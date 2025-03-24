//! This is a translation of embedding.cpp in llama.cpp using llama-cpp-2.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::fs;
use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{bail, Context, Result};

use llama_cpp::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp::context::LlamaContext;
use llama_cpp::ggml_time_us;
use llama_cpp::llama_backend::LlamaBackend;
use llama_cpp::llama_batch::LlamaBatch;
use llama_cpp::model::params::LlamaModelParams;
use llama_cpp::model::LlamaModel;
use llama_cpp::model::{AddBos, Special};
use llama_cpp_rs_bench::split_data::QuerySummaries;

fn main() -> Result<()> {
    // init LLM
    let backend = LlamaBackend::init()?;

    // offload all layers to the gpu
    let model_params = if cfg!(any(feature = "cuda", feature = "metal")) {
        LlamaModelParams::default().with_n_gpu_layers(1000)
    } else {
        LlamaModelParams::default()
    };

    //let model_path = PathBuf::from("models/bge-reranker-v2-m3-q4_k_m.gguf");
    let model_path = PathBuf::from("models/bge-reranker-v2-m3-f16.gguf");

    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .with_context(|| "unable to load model")?;
    // println!("pooling: {}", pooling);
    let pooling_type = LlamaPoolingType::Rank;

    println!("###### pooling_type: {:?}", pooling_type);

    let max_tokens = 4096;
    let ctx_params = LlamaContextParams::default()
        .with_n_threads_batch(std::thread::available_parallelism()?.get().try_into()?)
        .with_embeddings(true)
        .with_pooling_type(pooling_type)
        .with_n_ctx(NonZeroU32::new(max_tokens))
        .with_n_ubatch(max_tokens)
        .with_n_batch(max_tokens);
    println!("ctx_params: {:?}", ctx_params);
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create the llama_context")?;

    let n_embd = model.n_embd();

    let data_path = "tests/test_data/bert_paper_query_summaries.json";
    let input_str = fs::read_to_string(data_path)?;
    let query_summaries = serde_json::from_str::<QuerySummaries>(&input_str)?;
    let query = query_summaries.query;
    let documents = query_summaries.summaries;

    let eos = "</s>";
    let sep = "</s>";
    let bos = "<s>";

    let prompt_lines = {
        let mut lines = Vec::new();
        for doc in &documents {
            // Todo!  update to get eos and sep from model instead of hardcoding
            lines.push(format!("{bos}{query}{eos}{sep}{doc}{eos}"));
        }
        lines
    };

    // tokenize the prompt
    let tokens_lines_list = prompt_lines
        .iter()
        .map(|line| model.str_to_token(line, AddBos::Never))
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("failed to tokenize {:?}", prompt_lines))?;

    let n_ctx = ctx.n_ctx() as usize;

    if tokens_lines_list.iter().any(|tok| n_ctx < tok.len()) {
        bail!("One of the provided prompts exceeds the size of the context window");
    }

    // print the prompt token-by-token
    eprintln!();

    std::io::stderr().flush()?;

    // create a llama_batch with the size of the context
    // we use this object to submit token data for decoding
    let mut batch = LlamaBatch::new(max_tokens as usize, 1);

    // Todo!  update to get n_embd  to init vector size for better memory management
    // let mut n_embd_count = if pooling == "none" {
    //     tokens_lines_list.iter().map(|tokens| tokens.len()).sum()
    // } else {
    //     tokens_lines_list.len()
    // };
    let mut max_seq_id_batch = 0;
    let mut output = Vec::with_capacity(tokens_lines_list.len());

    let t_main_start = ggml_time_us();

    for tokens in &tokens_lines_list {
        // Flush the batch if the next prompt would exceed our batch size
        if (batch.n_tokens() as usize + tokens.len()) > max_tokens as usize {
            batch_decode(
                &mut ctx,
                &mut batch,
                max_seq_id_batch,
                &mut output,
                true,
                "rank".to_string(),
            )?;
            max_seq_id_batch = 0;
            batch.clear();
        }

        batch.add_sequence(tokens, max_seq_id_batch, false)?;
        max_seq_id_batch += 1;
    }
    // Handle final batch
    batch_decode(
        &mut ctx,
        &mut batch,
        max_seq_id_batch,
        &mut output,
        true,
        "rank".to_string(),
    )?;

    let t_main_end = ggml_time_us();

    let scores = output
        .iter()
        .map(|embeddings| embeddings[0])
        .collect::<Vec<f32>>();
    let mut scores = scores.iter().enumerate().collect::<Vec<(usize, &f32)>>();
    // sort by score
    scores.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (idx, score) in scores.iter() {
        println!("--------------- {} ---------------", idx);
        println!("score: {}", score);
        let summary = documents.get(*idx).unwrap();
        println!("summary: {}", summary);
    }

    let duration = Duration::from_micros((t_main_end - t_main_start) as u64);
    let total_tokens: usize = tokens_lines_list.iter().map(Vec::len).sum();
    eprintln!(
        "Created embeddings for {} tokens in {:.2} s, speed {:.2} t/s\n",
        total_tokens,
        duration.as_secs_f32(),
        total_tokens as f32 / duration.as_secs_f32()
    );

    println!("{}", ctx.timings());

    Ok(())
}

fn batch_decode(
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch,
    s_batch: i32,
    output: &mut Vec<Vec<f32>>,
    normalise: bool,
    pooling: String,
) -> Result<()> {
    eprintln!(
        "{}: n_tokens = {}, n_seq = {}",
        stringify!(batch_decode),
        batch.n_tokens(),
        s_batch
    );

    // Clear previous kv_cache values
    ctx.clear_kv_cache();

    ctx.decode(batch).with_context(|| "llama_decode() failed")?;

    for i in 0..s_batch {
        let embeddings = ctx
            .embeddings_seq_ith(i)
            .with_context(|| "Failed to get sequence embeddings")?;
        let normalized = if normalise {
            if pooling == "rank" {
                normalize_embeddings(&embeddings, -1)
            } else {
                normalize_embeddings(&embeddings, 2)
            }
        } else {
            embeddings.to_vec()
        };
        output.push(normalized);
    }

    batch.clear();

    Ok(())
}

/// Normalizes embeddings based on different normalization strategies
fn normalize_embeddings(input: &[f32], embd_norm: i32) -> Vec<f32> {
    let n = input.len();
    let mut output = vec![0.0; n];

    let sum = match embd_norm {
        -1 => 1.0, // no normalization
        0 => {
            // max absolute
            let max_abs = input.iter().map(|x| x.abs()).fold(0.0f32, f32::max) / 32760.0;
            max_abs as f64
        }
        2 => {
            // euclidean norm
            input
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum::<f64>()
                .sqrt()
        }
        p => {
            // p-norm
            let sum = input.iter().map(|x| (x.abs() as f64).powi(p)).sum::<f64>();
            sum.powf(1.0 / p as f64)
        }
    };

    let norm = if sum > 0.0 { 1.0 / sum } else { 0.0 };

    for i in 0..n {
        output[i] = (input[i] as f64 * norm) as f32;
    }

    output
}
