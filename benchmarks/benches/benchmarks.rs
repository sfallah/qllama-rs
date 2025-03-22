use criterion::{black_box, criterion_main, Criterion};
use llama_cpp::context::params::LlamaContextParams;
use llama_cpp::context::LlamaContext;
use llama_cpp::llama_backend::LlamaBackend;
use llama_cpp::model::params::LlamaModelParams;
use llama_cpp::model::LlamaModel;
use llama_cpp::token::LlamaToken;
use llama_cpp_rs_bench::{hf_tokenize, init_splitter, llama_cpp_tokenize, process_batch, process_single, to_llama_tokens};
use std::fs;
use std::num::NonZeroU32;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;
use fast_text_splitter::hf_tokenizer::init_tokenizer;
use rayon::prelude::*;
use tokenizers::Tokenizer;

pub fn llama_cpp_embedding(
    c: &mut Criterion,
    ctx: &mut LlamaContext,
    hf_tokens: &Vec<Vec<LlamaToken>>,
) {
    c.bench_function("llama_cpp_embedding_batch", |b| {
        b.iter(|| {
            let embeddings = process_batch(ctx, black_box(hf_tokens)).unwrap();
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

pub fn hf_tokenize_benchmark(c: &mut Criterion, hf_tokenizer: Arc<Tokenizer>, sentences: &Vec<String>) {
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
pub fn benches() {
    let mut criterion: Criterion<_> = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(20))
        .configure_from_args();

    let text_file = "tests/test_data/superlinear.txt";
    let data_str = fs::read_to_string(text_file).unwrap();
    let data = data_str.as_bytes();
    let mut backend = LlamaBackend::init().unwrap();
    //backend.void_logs();

    let model_params = if cfg!(any(feature = "cuda", feature = "metal")) {
        LlamaModelParams::default().with_n_gpu_layers(1000)
    } else {
        LlamaModelParams::default()
    };

    //let model_path = "models/gte-qwen2-1.5b-instruct-q4_k_m.gguf".to_string();
    let model_path = "models/snowflake-arctic-embed-m-v1.5-q4_k_m.gguf".to_string();
    //let model_path = "models/bge-large-en-v1.5-q4_k_m.gguf".to_string();
    let model_path = PathBuf::from(model_path);

    //let model_id = "Alibaba-NLP/gte-Qwen2-1.5B-instruct".to_string();
    let model_id = "Snowflake/snowflake-arctic-embed-m-v1.5".to_string();
    //let model_id = "BAAI/bge-large-en-v1.5".to_string();

    let model = LlamaModel::load_from_file(&backend, model_path, &model_params).unwrap();

    // initialize the context
    //let parallelism = std::thread::available_parallelism().unwrap().get() as u32;
    //println!("parallelism: {}", parallelism);
    let max_ctx = 3072;
    let ctx_params = LlamaContextParams::default()
        .with_embeddings(true)
        .with_n_batch(max_ctx)
        .with_n_ubatch(max_ctx)
        .with_n_ctx(NonZeroU32::new(max_ctx))
        .with_pooling_type(llama_cpp::context::params::LlamaPoolingType::Mean);

    let mut ctx = model.new_context(&backend, ctx_params).unwrap();

    let splitter_config = init_splitter(Some(model_id.clone()), None, Some(512), true).unwrap();
    let splits = splitter_config.hf_splits(data);
    println!("Number of splits: {:?}", splits.len());
    for (idx, split) in splits.iter().enumerate() {
        println!("split : {:?}", idx);
        println!("\tbytes : {:?}", split.split_string.len());
        println!("\ttokens: {:?}", split.tokens.len());
    }

    let tokens_list = splits
        .iter()
        .map(|res| llama_cpp_tokenize(&model, res.split_string.as_str()))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    /*
    let hf_tokens: Vec<_> = splits
        .iter()
        .map(|split| to_llama_tokens(&split.tokens, &ctx.model).unwrap())
        .collect();

     */

    llama_cpp_tokenize_benchmark(
        &mut criterion,
        &model,
        &splits
            .iter()
            .map(|split| split.split_string.clone())
            .collect(),
    );

    let hf_tokenizer:Tokenizer = init_tokenizer(Some(model_id.clone()), None, true).unwrap();
    let hf_tokenizer = Arc::new(hf_tokenizer);
    hf_tokenize_benchmark(
        &mut criterion,
        hf_tokenizer.clone(),
        &splits
            .iter()
            .map(|split| split.split_string.clone())
            .collect(),
    );

    hf_parallel_tokenize_benchmark(
        &mut criterion,
        hf_tokenizer.clone(),
        &splits
            .iter()
            .map(|split| split.split_string.clone())
            .collect(),
    );

    llama_cpp_embedding(&mut criterion, &mut ctx, &tokens_list);
    //llama_cpp_embedding_single(&mut criterion, &mut ctx, &hf_tokens);
}

criterion_main!(benches);
