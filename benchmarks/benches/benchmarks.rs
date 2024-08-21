use criterion::{black_box, criterion_main, Criterion};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_rs_bench::{get_splitter_config, process_batch, to_llama_tokens};

pub fn llama_cpp_embedding(c: &mut Criterion) {
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
    let model_path = PathBuf::from(model_path);

    let model = LlamaModel::load_from_file(&backend, model_path, &model_params).unwrap();

    // initialize the context
    let ctx_params_default = LlamaContextParams::default();
    let ctx_params = LlamaContextParams::default().with_n_threads_batch(std::thread::available_parallelism().unwrap().get() as u32)
        .with_n_ubatch(ctx_params_default.n_batch())
        .with_embeddings(true);

    let mut ctx = model.new_context(&backend, ctx_params).unwrap();
    let n_ctx = ctx.n_ctx() as usize;

    let splitter_config = get_splitter_config(None, Some(n_ctx - 2)).unwrap();
    let splits = splitter_config.hf_splits(data);
    let hf_tokens: Vec<_> = splits.iter().map(|split| to_llama_tokens(&split.tokens, &ctx.model).unwrap()).collect();


    c.bench_function("candle_bert_embedding", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                let embeddings = process_batch(&mut ctx, &hf_tokens).unwrap();
                black_box(&embeddings);
            }
            start.elapsed()
        });
    });
}

pub fn benches() {
    let mut criterion: Criterion<_> = Criterion::default()
        .sample_size(40)
        .measurement_time(std::time::Duration::from_secs(10))
        .configure_from_args();

    llama_cpp_embedding(&mut criterion);
}

criterion_main!(benches);
