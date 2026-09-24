use anyhow::Context;
use criterion::{criterion_main, Criterion};
use qllama::context::params::{LlamaContextParams, LlamaPoolingType};
use qllama::context::LlamaContext;
use qllama::model::{AddBos, LlamaModel};
use qllama::token::LlamaToken;
use qllama_bench::split_data::QuerySummaries;
use qllama_bench::{
    ensure_hf_model_file, hf_tokenize, init_backend, init_model, init_reranker_context,
    init_splitter, llama_cpp_tokenize, load_query_summaries, process_batch, process_single,
    rerank_token_batches,
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

/// Qwen3-Reranker prompt, as stored in the GGUF's `tokenizer.chat_template.rerank`.
fn qwen3_rerank_prompt(query: &str, document: &str) -> String {
    format!(
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {query}\n<Document>: {document}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
}

fn qwen3_rerank_tokens(
    model: &LlamaModel,
    query_summaries: &QuerySummaries,
) -> Vec<Vec<LlamaToken>> {
    query_summaries
        .summaries
        .iter()
        .map(|doc| {
            model.str_to_token(
                &qwen3_rerank_prompt(&query_summaries.query, doc),
                AddBos::Never,
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

/// Rank pooling on Qwen3 runs the `cls.output` yes/no head on the last token and softmaxes it,
/// so element 0 of each sequence's output is P(yes).
fn qwen3_rerank_scores(
    ctx: &mut LlamaContext,
    tokens: &[Vec<LlamaToken>],
    max_tokens: u32,
) -> Vec<f32> {
    rerank_token_batches(ctx, tokens, max_tokens as usize, false, "rank")
        .unwrap()
        .iter()
        .map(|output| output[0])
        .collect()
}

/// End to end: prompt formatting, tokenisation and ranking of all documents.
pub fn qwen3_reranker(
    c: &mut Criterion,
    ctx: &mut LlamaContext,
    model: &LlamaModel,
    query_summaries: &QuerySummaries,
    max_tokens: u32,
) {
    c.bench_function("qwen3_reranker", |b| {
        b.iter(|| {
            let tokens = qwen3_rerank_tokens(model, black_box(query_summaries));
            black_box(qwen3_rerank_scores(ctx, &tokens, max_tokens));
        });
    });
}

/// Ranking only, on prompts tokenised once up front.
pub fn qwen3_reranker_pretokenized(
    c: &mut Criterion,
    ctx: &mut LlamaContext,
    model: &LlamaModel,
    query_summaries: &QuerySummaries,
    max_tokens: u32,
) {
    let tokens = qwen3_rerank_tokens(model, query_summaries);
    c.bench_function("qwen3_reranker_pretokenized", |b| {
        b.iter(|| black_box(qwen3_rerank_scores(ctx, black_box(&tokens), max_tokens)));
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
    llama_cpp_embedding_sentences(&mut criterion, &mut ctx, &model, &sentences1, &sentences2);

    let query_summaries =
        load_query_summaries("tests/test_data/bert_paper_query_summaries.json").unwrap();

    let model_path = ensure_hf_model_file(
        "sabafallah/Qwen3-Reranker-0.6B-Q4_K_M-GGUF",
        "qwen3-reranker-0.6b-q4_k_m.gguf",
        None,
    )
    .unwrap();
    let model = init_model(&model_path, &backend).unwrap();
    let max_tokens = 2048;
    let mut ctx =
        init_reranker_context(&model, &backend, max_tokens, Some(LlamaPoolingType::Rank)).unwrap();

    // Sanity check before timing: the ranking should put BERT-method summaries on top.
    let tokens = qwen3_rerank_tokens(&model, &query_summaries);
    let mut ranked: Vec<(usize, f32)> = qwen3_rerank_scores(&mut ctx, &tokens, max_tokens)
        .into_iter()
        .enumerate()
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("query: {}", query_summaries.query);
    for (idx, score) in &ranked {
        let summary = &query_summaries.summaries[*idx];
        let head: String = summary
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .take(90)
            .collect();
        println!(
            "{score:.4}  [{idx:2}] {} tokens  {head}",
            tokens[*idx].len()
        );
    }

    qwen3_reranker(
        &mut criterion,
        &mut ctx,
        &model,
        &query_summaries,
        max_tokens,
    );
    qwen3_reranker_pretokenized(
        &mut criterion,
        &mut ctx,
        &model,
        &query_summaries,
        max_tokens,
    );
}

criterion_main!(benches);
