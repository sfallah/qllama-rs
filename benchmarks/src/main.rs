//! This is a translation of embedding.cpp in llama.cpp using llama-cpp-2.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::time::Duration;

use anyhow::{bail, Context, Result};

use llama_cpp::context::params::LlamaPoolingType;
use llama_cpp::ggml_time_us;
use llama_cpp::model::AddBos;
use llama_cpp_rs_bench::{
    build_simple_reranker_prompts, ensure_hf_model_file, init_backend, init_model,
    init_reranker_context, load_query_summaries, rerank_token_batches,
};

fn main() -> Result<()> {
    // init LLM
    let backend = init_backend(true)?;

    //let model_path = "models/bge-reranker-v2-m3-q4_k_m.gguf";
    //let model_path = "models/jina-reranker-v1-tiny-en-q4_k_m.gguf";
    //let model_path = "models/jina-reranker-v1-tiny-en-FP16.gguf";
    let model_path = ensure_hf_model_file(
        "sabafallah/bge-reranker-base-Q4_K_M-GGUF",
        "bge-reranker-base-q4_k_m.gguf",
        None,
    )?;
    //let model_path = "models/bge-m3-q4_k_m.gguf";
    //let model_path = "models/bge-reranker-v2-m3-f16.gguf";

    let model = init_model(&model_path, &backend).with_context(|| "unable to load model")?;
    let pooling_type = LlamaPoolingType::Rank;

    println!("###### pooling_type: {:?}", pooling_type);

    let max_tokens = 4096;
    let mut ctx = init_reranker_context(&model, &backend, max_tokens, Some(pooling_type))
        .with_context(|| "unable to create the llama_context")?;

    let _n_embd = model.n_embd();

    let data_path = "tests/test_data/bert_paper_query_summaries.json";
    let query_summaries = load_query_summaries(data_path)?;
    let query = query_summaries.query;
    let documents = query_summaries.summaries;

    let prompt_lines = build_simple_reranker_prompts(&query, &documents);

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

    let t_main_start = ggml_time_us();

    let output = rerank_token_batches(
        &mut ctx,
        &tokens_lines_list,
        max_tokens as usize,
        true,
        "rank",
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
