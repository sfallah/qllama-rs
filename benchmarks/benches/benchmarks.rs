use anyhow::Context;
use criterion::{criterion_main, Criterion};
use indexmap::IndexMap;
use qllama::context::params::{LlamaContextParams, LlamaPoolingType};
use qllama::context::LlamaContext;
use qllama::llama_batch::LlamaBatch;
use qllama::model::{AddBos, LlamaModel};
use qllama::token::LlamaToken;
use qllama_bench::split_data::QuerySummaries;
use qllama_bench::{
    batch_decode_rerank, build_simple_reranker_prompts, ensure_hf_model_file, hf_tokenize,
    init_backend, init_model, init_reranker_context, init_splitter, llama_cpp_tokenize,
    load_query_summaries, process_batch, process_single, rerank_token_batches,
};
use rayon::prelude::*;
use std::fs;
use std::hint::black_box;
use std::num::NonZeroU32;
use std::sync::Arc;
use tokenizers::Tokenizer;

pub fn llama_cpp_embedding(
    c: &mut Criterion,
    ctx: &mut LlamaContext,
    model: &LlamaModel,
    splits: &[String],
) {
    c.bench_function("llama_cpp_embedding_batch", |b| {
        b.iter(|| {
            let tokens_list = splits
                .iter()
                .map(|res| llama_cpp_tokenize(model, res.as_str()))
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
    sentences1: &[&str],
    sentences2: &[&str],
) {
    c.bench_function("llama_cpp_embedding_sentences", |b| {
        b.iter(|| {
            let tokens_list = sentences1
                .iter()
                .map(|res| llama_cpp_tokenize(model, res))
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let embeddings = process_batch(ctx, black_box(&tokens_list)).unwrap();
            black_box(embeddings);

            let tokens_list = sentences2
                .iter()
                .map(|res| llama_cpp_tokenize(model, res))
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
    sentences: &[String],
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
    sentences: &[String],
) {
    c.bench_function("hf_tokenize_benchmark", |b| {
        b.iter(|| {
            let tokens_list = sentences
                .iter()
                .map(|sentence| hf_tokenize(&hf_tokenizer, sentence))
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            black_box(tokens_list);
        });
    });
}

pub fn hf_parallel_tokenize_benchmark(
    c: &mut Criterion,
    hf_tokenizer: Arc<Tokenizer>,
    sentences: &[String],
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
    hf_tokens: &[Vec<LlamaToken>],
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
    let prompt_lines = build_simple_reranker_prompts(query, documents);

    c.bench_function("reranker_benchmark", |b| {
        b.iter(|| {
            let tokens_lines_list = prompt_lines
                .iter()
                .map(|line| model.str_to_token(line, AddBos::Never))
                .collect::<Result<Vec<_>, _>>()
                .with_context(|| format!("failed to tokenize {:?}", prompt_lines))
                .unwrap();

            let output =
                rerank_token_batches(ctx, &tokens_lines_list, max_tokens as usize, true, "rank")
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
            let query_tokens = model
                .str_to_token(query, AddBos::Never)
                .unwrap_or_else(|e| panic!("Failed to tokenize query: {:?}", e));
            let query_no_tokens = query_tokens.len();
            let n_ctx = ctx.n_ctx() as usize;

            let mut sequence_pairs_map = IndexMap::new();
            for (idx, text) in texts.iter().enumerate() {
                let text_tokens = model
                    .str_to_token(text, AddBos::Never)
                    .unwrap_or_else(|e| panic!("Failed to tokenize seq: {}\n, text: {:?}\n, error: {:?}", idx, text, e));
                let text_no_tokens = text_tokens.len();
                if text_no_tokens + query_no_tokens + 4 > n_ctx {
                    panic!(
                        "Sequence Pair no_tokens exceeds n_ctx. Query: {}, text: {}, n_ctx: {}",
                        query_no_tokens, text_no_tokens, n_ctx
                    );
                }
                //"{bos}{query}{eos}{sep}{doc}{eos}"
                let mut sequence_pairs_tokens = query_tokens.clone();
                sequence_pairs_tokens.insert(0, bos_token);
                sequence_pairs_tokens.push(eos_token);
                sequence_pairs_tokens.push(sep_token);
                sequence_pairs_tokens.extend(text_tokens);
                sequence_pairs_tokens.push(eos_token);
                sequence_pairs_map.insert(idx, sequence_pairs_tokens);
            }

            let mut batch = LlamaBatch::new(max_tokens as usize, 1);
            let mut max_seq_id_batch = 0;
            let mut output = Vec::with_capacity(sequence_pairs_map.len());

            for tokens in sequence_pairs_map.values() {
                // Flush the batch if the next prompt would exceed our batch size
                if batch.n_tokens() as usize + tokens.len() > n_ctx {
                    batch_decode_rerank(ctx, &mut batch, max_seq_id_batch, &mut output, true, "rank").unwrap();
                    max_seq_id_batch = 0;
                    batch.clear();
                }
                batch.add_sequence(tokens, max_seq_id_batch, false).expect("Failed to add sequence to batch");
                max_seq_id_batch += 1;
            }

            batch_decode_rerank(ctx, &mut batch, max_seq_id_batch, &mut output, true, "rank").unwrap();

            let scores = output.iter().map(|embeddings| embeddings[0]).collect::<Vec<f32>>();
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
    let data_str = fs::read_to_string(text_file).unwrap();
    let data = data_str.as_bytes();
    let backend = init_backend(false).unwrap();

    let model_path = ensure_hf_model_file(
        "sabafallah/all-MiniLM-L6-v2-Q4_K_M-GGUF",
        "all-minilm-l6-v2-q4_k_m.gguf",
        None,
    )
    .unwrap();

    let model = init_model(&model_path, &backend).unwrap();

    let max_ctx = 512;
    let ctx_params = LlamaContextParams::default()
        .with_embeddings(true)
        .with_n_batch(max_ctx)
        .with_n_ubatch(max_ctx)
        .with_n_threads(2)
        .with_kv_unified(true)
        .with_n_ctx(NonZeroU32::new(max_ctx));

    let mut ctx = model.new_context(&backend, ctx_params).unwrap();

    let model_id = "sentence-transformers/all-MiniLM-L6-v2";
    let splitter_config = init_splitter(Some(model_id.to_string()), None, Some(400), true).unwrap();
    let splits = splitter_config.hf_splits(data);
    println!("Number of splits: {:?}", splits.len());
    for (idx, split) in splits.iter().enumerate() {
        println!("split : {:?}", idx);
        println!("\tbytes : {:?}", split.split_string.len());
        println!("\ttokens: {:?}", split.tokens.len());
    }

    let splits_str: Vec<String> = splits.iter().map(|split| split.split_string.clone()).collect();
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
    llama_cpp_embedding_sentences(&mut criterion, &mut ctx, &model, &sentences1, &sentences2);

    let _query_summaries =
        load_query_summaries("tests/test_data/bert_paper_query_summaries.json").unwrap();

    let model_path = ensure_hf_model_file(
        "sabafallah/Qwen3-Reranker-0.6B-Q4_K_M-GGUF",
        "qwen3-reranker-0.6b-q4_k_m.gguf",
        None,
    )
    .unwrap();
    let model = init_model(&model_path, &backend).unwrap();
    let max_tokens = 2048;
    let _ctx =
        init_reranker_context(&model, &backend, max_tokens, Some(LlamaPoolingType::Last)).unwrap();

    /*
    reranker_benchmark(&mut criterion, &mut ctx, &model, &query_summaries, max_tokens);
    reranker_benchmark_improved(&mut criterion, &mut ctx, &model, &query_summaries, max_tokens);
    */
}

criterion_main!(benches);
