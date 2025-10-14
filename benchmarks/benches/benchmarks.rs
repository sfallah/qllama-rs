use anyhow::Context;
use criterion::{criterion_main, Criterion};
use indexmap::IndexMap;
use llama_cpp::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp::context::LlamaContext;
use llama_cpp::llama_backend::LlamaBackend;
use llama_cpp::llama_batch::LlamaBatch;
use llama_cpp::model::params::LlamaModelParams;
use llama_cpp::model::{AddBos, LlamaModel};
use llama_cpp::token::LlamaToken;
use llama_cpp_rs_bench::split_data::QuerySummaries;
use llama_cpp_rs_bench::{
    batch_decode_rerank, hf_tokenize, init_model, init_reranker_context, init_splitter,
    llama_cpp_tokenize, process_batch, process_single, to_llama_tokens,
};
use rayon::prelude::*;
use std::fs;
use std::hint::black_box;
use std::num::NonZeroU32;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

pub fn llama_cpp_embedding(
    c: &mut Criterion,
    ctx: &mut LlamaContext,
    model: &LlamaModel,
    splits: &Vec<String>,
) {
    c.bench_function("llama_cpp_embedding_batch", |b| {
        b.iter(|| {
            let tokens_list = splits
                .iter()
                .map(|res| llama_cpp_tokenize(&model, res.as_str()))
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let embeddings = process_batch(ctx, black_box(&tokens_list)).unwrap();
            black_box(embeddings);
        });
    });
}

pub fn llama_cpp_embedding_sentences(
    c: &mut Criterion,
    ctx: &mut LlamaContext,
    model: &LlamaModel,
    sentences1: &Vec<&str>,
    sentences2: &Vec<&str>,
) {
    c.bench_function("llama_cpp_embedding_sentences", |b| {
        b.iter(|| {
            let tokens_list = sentences1
                .iter()
                .map(|res| llama_cpp_tokenize(&model, res))
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let embeddings = process_batch(ctx, black_box(&tokens_list)).unwrap();
            black_box(embeddings);

            let tokens_list = sentences2
                .iter()
                .map(|res| llama_cpp_tokenize(&model, res))
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let embeddings = process_batch(ctx, black_box(&tokens_list)).unwrap();
            black_box(embeddings);
        });
    });
}

pub fn llama_cpp_tokenize_benchmark(
    c: &mut Criterion,
    model: &LlamaModel,
    sentences: &Vec<String>,
) {
    c.bench_function("llama_cpp_tokenize_benchmark", |b| {
        b.iter(|| {
            let tokens_list = sentences
                .iter()
                .map(|sentence| llama_cpp_tokenize(model, sentence))
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            black_box(tokens_list);
        });
    });
}

pub fn hf_tokenize_benchmark(
    c: &mut Criterion,
    hf_tokenizer: Arc<Tokenizer>,
    sentences: &Vec<String>,
) {
    c.bench_function("hf_tokenize_benchmark", |b| {
        b.iter(|| {
            let tokens_list = sentences
                .iter()
                .map(|sentence| hf_tokenize(hf_tokenizer.deref(), sentence))
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            black_box(tokens_list);
        });
    });
}

pub fn hf_parallel_tokenize_benchmark(
    c: &mut Criterion,
    hf_tokenizer: Arc<Tokenizer>,
    sentences: &Vec<String>,
) {
    c.bench_function("hf_parallel_tokenize_benchmark", |b| {
        b.iter(|| {
            let tokens_list = sentences
                .par_iter()
                .map(|sentence| hf_tokenize(&hf_tokenizer.clone(), sentence.as_str()))
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            black_box(tokens_list);
        });
    });
}

pub fn llama_cpp_embedding_single(
    c: &mut Criterion,
    ctx: &mut LlamaContext,
    hf_tokens: &Vec<Vec<LlamaToken>>,
) {
    c.bench_function("llama_cpp_embedding_single", |b| {
        b.iter(|| {
            black_box(hf_tokens).iter().for_each(|hf_tokens| {
                let embedding = process_single(ctx, black_box(hf_tokens)).unwrap();
                black_box(embedding);
            });
        });
    });
}

pub fn reranker_benchmark(
    c: &mut Criterion,
    ctx: &mut LlamaContext,
    model: &LlamaModel,
    query_summaries: &QuerySummaries,
    max_tokens: u32,
) {
    let query = &query_summaries.query;
    let documents = &query_summaries.summaries;

    let eos = "</s>";
    let sep = "</s>";
    let bos = "<s>";

    let prompt_lines = {
        let mut lines = Vec::new();
        for doc in documents {
            // Todo!  update to get eos and sep from model instead of hardcoding
            lines.push(format!("{bos}{query}{eos}{sep}{doc}{eos}"));
        }
        lines
    };

    c.bench_function("reranker_benchmark", |b| {
        b.iter(|| {
            let tokens_lines_list = prompt_lines
                .iter()
                .map(|line| model.str_to_token(line, AddBos::Never))
                .collect::<anyhow::Result<Vec<_>, _>>()
                .with_context(|| format!("failed to tokenize {:?}", prompt_lines))
                .unwrap();

            let n_ctx = ctx.n_ctx() as usize;
            let mut batch = LlamaBatch::new(max_tokens as usize, 0, 1);

            let mut max_seq_id_batch = 0;
            let mut output = Vec::with_capacity(tokens_lines_list.len());
            let normalise = true;
            for tokens in &tokens_lines_list {
                // Flush the batch if the next prompt would exceed our batch size
                if (batch.n_tokens() as usize + tokens.len()) > max_tokens as usize {
                    batch_decode_rerank(
                        ctx,
                        &mut batch,
                        max_seq_id_batch,
                        &mut output,
                        normalise,
                        "rank".to_string(),
                    )
                    .unwrap();
                    max_seq_id_batch = 0;
                    batch.clear();
                }

                batch.add_sequence(tokens, max_seq_id_batch, false).unwrap();
                max_seq_id_batch += 1;
            }
            // Handle final batch
            batch_decode_rerank(
                ctx,
                &mut batch,
                max_seq_id_batch,
                &mut output,
                normalise,
                "rank".to_string(),
            )
            .unwrap();

            let scores = output
                .iter()
                .map(|embeddings| embeddings[0])
                .collect::<Vec<f32>>();
            black_box(scores);
        });
    });
}

pub fn reranker_benchmark_improved(
    c: &mut Criterion,
    ctx: &mut LlamaContext,
    model: &LlamaModel,
    query_summaries: &QuerySummaries,
    max_tokens: u32,
) {
    let query = &query_summaries.query;
    let texts = &query_summaries.summaries;

    let bos_token = model.token_bos();
    let eos_token = model.token_eos();
    let sep_token = model.token_sep();

    c.bench_function("reranker_benchmark_improved", |b| {
        b.iter(|| {
            let query_tokens = match model.str_to_token(&query, AddBos::Never) {
                Ok(tokens) => tokens,
                Err(e) => {
                    let error_msg = format!("Failed to tokenize query: {:?}", e);
                    eprintln!("{}", error_msg);
                    panic!("{}", error_msg);
                }
            };
            let query_no_tokens = query_tokens.len();
            let n_ctx = ctx.n_ctx() as usize;

            let mut sequence_pairs_map = IndexMap::new();
            for (idx, text) in texts.iter().enumerate() {
                let text_tokens = match model.str_to_token(text, AddBos::Never) {
                    Ok(tokens) => tokens,
                    Err(e) => {
                        let error_msg = format!(
                            "Failed to tokenize seq: {}\n, text: {:?}\n, error: {:?}",
                            idx, text, e
                        );
                        eprintln!("{}", error_msg);
                        panic!("{}", error_msg);
                    }
                };
                let text_no_tokens = text_tokens.len();
                if text_no_tokens + query_no_tokens + 4 > n_ctx as usize {
                    let error_msg = format!(
                        "Sequence Pair no_tokens exceeds n_ctx. Query: {}, text: {}, n_ctx: {}",
                        query_no_tokens, text_no_tokens, n_ctx
                    );
                    eprintln!("{}", error_msg);
                    panic!("{}", error_msg);
                }
                //"{bos}{query}{eos}{sep}{doc}{eos}"
                let mut sequence_pairs_tokens = query_tokens.clone();
                sequence_pairs_tokens.insert(0, bos_token);
                sequence_pairs_tokens.push(eos_token);
                sequence_pairs_tokens.push(sep_token);
                sequence_pairs_tokens.append(&mut text_tokens.clone());
                sequence_pairs_tokens.push(eos_token);
                sequence_pairs_map.insert(idx, sequence_pairs_tokens);
            }

            let n_ctx = ctx.n_ctx() as usize;
            let mut batch = LlamaBatch::new(max_tokens as usize, 0, 1);

            let mut max_seq_id_batch = 0;
            let mut output = Vec::with_capacity(sequence_pairs_map.len());

            for tokens in sequence_pairs_map.values().into_iter() {
                // Flush the batch if the next prompt would exceed our batch size
                if (batch.n_tokens() as usize + tokens.len()) > n_ctx as usize {
                    batch_decode_rerank(
                        ctx,
                        &mut batch,
                        max_seq_id_batch,
                        &mut output,
                        true,
                        "rank".to_string(),
                    )
                    .unwrap();
                    max_seq_id_batch = 0;
                    batch.clear();
                }

                batch
                    .add_sequence(tokens, max_seq_id_batch, false)
                    .expect("Failed to add sequence to batch");
                max_seq_id_batch += 1;
            }

            batch_decode_rerank(
                ctx,
                &mut batch,
                max_seq_id_batch,
                &mut output,
                true,
                "rank".to_string(),
            )
            .unwrap();

            let scores = output
                .iter()
                .map(|embeddings| embeddings[0])
                .collect::<Vec<f32>>();
            black_box(scores)
        });
    });
}
pub fn benches() {
    let mut criterion: Criterion<_> = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(20))
        .configure_from_args();

    let text_file = "tests/test_data/superlinear.txt";
    //let text_file = "tests/test_data/bert_paper.txt";
    let data_str = fs::read_to_string(text_file).unwrap();
    let data = data_str.as_bytes();
    let mut backend = LlamaBackend::init().unwrap();
    backend.void_logs();

    let model_params = if cfg!(any(
        feature = "cuda",
        feature = "cuda-no-vmm",
        feature = "metal"
    )) {
        LlamaModelParams::default().with_n_gpu_layers(1000)
    } else {
        LlamaModelParams::default()
    };

    //let model_path = "models/gte-qwen2-1.5b-instruct-q4_k_m.gguf".to_string();
    //let model_path = "models/snowflake-arctic-embed-m-v1.5-q4_k_m.gguf".to_string();
    //let model_path = "models/bge-large-en-v1.5-q4_k_m.gguf".to_string();
    //let model_path = "models/all-MiniLM-L6-v2-Q4_K_M.gguf".to_string();
    //let model_path = "models/Qwen3-Embedding-0.6B-Q8_0.gguf".to_string();
    //let model_path = "models/qwen3-embedding-0.6b-q4_k_m.gguf".to_string();
    //let model_path = "models/embeddinggemma-300m.gguf".to_string();
    //let model_path = "models/google/embeddinggemma-300m-Q4_K_M.gguf".to_string();
    //let model_path = "models/embeddinggemma-300m-qat-q4_0-unquantized_with_st_denses.gguf".to_string();
    //let model_path = "models/embeddinggemma-300m-qat-q4_0-unquantized_with_st_denses-Q4_0.gguf".to_string();
    let model_path = "models/embeddinggemma-300m-sentence-transformers-q8_0.gguf".to_string();
    //let model_path = "models/embeddinggemma-300m-sentence-transformers-qat-q4_0.gguf".to_string();
    //let model_path = "models/embeddinggemma-300m-sentence-transformers-qat-q8_0.gguf".to_string();

    let model_path = PathBuf::from(model_path);

    //let model_id = "Alibaba-NLP/gte-Qwen2-1.5B-instruct".to_string();
    //let model_id = "Snowflake/snowflake-arctic-embed-m-v1.5".to_string();
    //let model_id = "BAAI/bge-large-en-v1.5".to_string();
    //let model_id = "sentence-transformers/all-MiniLM-L6-v2".to_string();
    //let model_id = "Qwen/Qwen3-Embedding-0.6B".to_string();
    //let model_id = "google/embeddinggemma-300m".to_string();
    let model_id = "google/embeddinggemma-300m-qat-q4_0-unquantized".to_string();

    let model = LlamaModel::load_from_file(&backend, model_path, &model_params).unwrap();

    // initialize the context
    //let parallelism = std::thread::available_parallelism().unwrap().get() as u32;
    //println!("parallelism: {}", parallelism);
    let max_ctx = 2048;
    let ctx_params = LlamaContextParams::default()
        .with_embeddings(true)
        .with_n_batch(max_ctx)
        .with_n_ubatch(max_ctx)
        .with_n_ctx(NonZeroU32::new(max_ctx))
        .with_pooling_type(llama_cpp::context::params::LlamaPoolingType::Mean)
        .with_kv_unified(true);

    let mut ctx = model.new_context(&backend, ctx_params).unwrap();

    let splitter_config = init_splitter(Some(model_id.clone()), None, Some(400), true).unwrap();
    let splits = splitter_config.hf_splits(data);
    println!("Number of splits: {:?}", splits.len());
    for (idx, split) in splits.iter().enumerate() {
        println!("split : {:?}", idx);
        println!("\tbytes : {:?}", split.split_string.len());
        println!("\ttokens: {:?}", split.tokens.len());
    }

    /*
    let hf_tokens: Vec<_> = splits
        .iter()
        .map(|split| to_llama_tokens(&split.tokens, &ctx.model).unwrap())
        .collect();

     */

    /*
    llama_cpp_tokenize_benchmark(
        &mut criterion,
        &model,
        &splits
            .iter()
            .map(|split| split.split_string.clone())
            .collect(),
    );

     */

    //let hf_tokenizer:Tokenizer = init_tokenizer(Some(model_id.clone()), None, true).unwrap();
    //let hf_tokenizer = Arc::new(hf_tokenizer);
    /*
    hf_tokenize_benchmark(
        &mut criterion,
        hf_tokenizer.clone(),
        &splits
            .iter()
            .map(|split| split.split_string.clone())
            .collect(),
    );

     */

    /*
    hf_parallel_tokenize_benchmark(
        &mut criterion,
        hf_tokenizer.clone(),
        &splits
            .iter()
            .map(|split| split.split_string.clone())
            .collect(),
    );

     */

    let splits_str: Vec<String> = splits
        .iter()
        .map(|split| split.split_string.clone())
        .collect();

    llama_cpp_embedding(&mut criterion, &mut ctx, &model, &splits_str);

    let sentences1 = vec![
        "The new movie is awesome",
        "The cat sits outside",
        "A man is playing guitar",
        "I love pasta",
    ];

    let sentences2 = vec![
        "The dog plays in the garden",
        "The new movie is so great",
        "A woman watches TV",
        "Do you like pizza?",
    ];

    llama_cpp_embedding_sentences(
        &mut criterion,
        &mut ctx,
        &model,
        &sentences1,
        &sentences2,
    );

    let json_file_path = "tests/test_data/bert_paper_query_summaries.json";
    let input_str = fs::read_to_string(json_file_path).unwrap();
    let query_summaries = serde_json::from_str::<QuerySummaries>(&input_str).unwrap();

    //let model_path = "models/bge-reranker-v2-m3-f16.gguf";
    //let model_path = "models/bge-reranker-v2-m3-q4_k_m.gguf";
    //let model_path = "models/jina-reranker-v1-tiny-en-q4_k_m.gguf";
    let model_path = "models/qwen3-reranker-0.6b-q4_k_m.gguf";
    let model = init_model(model_path, &backend).unwrap();
    let max_tokens = 2048;
    let mut ctx =
        init_reranker_context(&model, &backend, max_tokens, Some(LlamaPoolingType::Last)).unwrap();

    /*
    reranker_benchmark(
        &mut criterion,
        &mut ctx,
        &model,
        &query_summaries,
        max_tokens,
    );
     */

    /*

    reranker_benchmark_improved(
        &mut criterion,
        &mut ctx,
        &model,
        &query_summaries,
        max_tokens,
    );

     */
}

criterion_main!(benches);
