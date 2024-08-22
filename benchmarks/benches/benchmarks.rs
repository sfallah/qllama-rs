use criterion::{black_box, criterion_main, Criterion};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use candle_core::Tensor;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_rs_bench::{get_splitter_config, process_batch, process_single, process_splits_batch, to_llama_tokens};

pub fn llama_cpp_embedding(c: &mut Criterion, ctx: &mut LlamaContext, hf_tokens: &Vec<Vec<LlamaToken>>) {
    c.bench_function("llama_cpp_embedding_batch", |b| {
        b.iter(|| {
                let embeddings = process_batch(ctx, black_box(hf_tokens)).unwrap();
                black_box(embeddings);
        });
    });
}

pub fn llama_cpp_embedding_tokenize(c: &mut Criterion, model: &LlamaModel, ctx: &mut LlamaContext, sentences: &Vec<String>) {
    c.bench_function("llama_cpp_embedding_tokenize", |b| {
        b.iter(|| {
            let embeddings = process_splits_batch(model, ctx, black_box(sentences)).unwrap();
            black_box(embeddings);
        });
    });
}

pub fn llama_cpp_embedding_single(c: &mut Criterion, ctx: &mut LlamaContext, hf_tokens: &Vec<Vec<LlamaToken>>) {
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
        .sample_size(40)
        .measurement_time(std::time::Duration::from_secs(10))
        .configure_from_args();

    let text_file = "tests/test_data/superlinear.txt";
    let data_str = fs::read_to_string(text_file).unwrap();
    let data = data_str.as_bytes();
    let backend = LlamaBackend::init().unwrap();
    //backend.void_logs();

    let mut model_params = if cfg!(any(feature = "cuda", feature = "vulkan", feature = "metal")) {
        LlamaModelParams::default().with_n_gpu_layers(1000)
    } else {
        LlamaModelParams::default()
    };

    let model_path = "/Users/sabafallah/dev/qimia_ai_dev/llama.cpp/models/all-MiniLM-L6-v2.gguf".to_string();
    //let model_path = "/Users/sabafallah/dev/qimia_ai_dev/llama.cpp/models/multilingual-e5-large-instruct-q4_k_m.gguf".to_string();
    let model_path = PathBuf::from(model_path);

    let model = LlamaModel::load_from_file(&backend, model_path, &model_params).unwrap();

    // initialize the context
    let ctx_params_default = LlamaContextParams::default();
    let parallelism = std::thread::available_parallelism().unwrap().get() as u32;
    println!("parallelism: {}", parallelism);
    let ctx_params = LlamaContextParams::default().with_n_threads_batch(parallelism)
        .with_n_ubatch(ctx_params_default.n_batch())
        .with_embeddings(true);

    let mut ctx = model.new_context(&backend, ctx_params).unwrap();

    //let splitter_config = get_splitter_config(Some("intfloat/multilingual-e5-large-instruct".to_string()), Some(512)).unwrap();
    let splitter_config = get_splitter_config(None, Some(512)).unwrap();
    let splits = splitter_config.hf_splits(data);
    let hf_tokens: Vec<_> = splits.iter().map(|split| to_llama_tokens(&split.tokens, &ctx.model).unwrap()).collect();

    llama_cpp_embedding(&mut criterion, &mut ctx, &hf_tokens);
    llama_cpp_embedding_tokenize(&mut criterion, &model, &mut ctx, &splits.iter().map(|split| split.split_string.clone()).collect());
    llama_cpp_embedding_single(&mut criterion, &mut ctx, &hf_tokens);
}

criterion_main!(benches);
