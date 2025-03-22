pub mod split_data;

use anyhow::{Context, Result};
use fast_text_splitter::config::SplitterLiteConfig;
use fast_text_splitter::hf_tokenizer::HFTokenizer;
use llama_cpp::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp::context::LlamaContext;
use llama_cpp::llama_backend::LlamaBackend;
use llama_cpp::llama_batch::LlamaBatch;
use llama_cpp::model::params::LlamaModelParams;
use llama_cpp::model::{AddBos, LlamaModel};
use llama_cpp::token::LlamaToken;
use std::fmt::Debug;
use std::num::NonZeroU32;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Clone, PartialEq)]
pub struct SentenceScore {
    pub sentence: String,
    pub score: f32,
}

impl Debug for SentenceScore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {:.16}", self.sentence, self.score)
    }
}

#[derive(Debug, Clone)]
pub struct SentenceSimilarity {
    pub sentence: String,
    pub similarities: Vec<SentenceScore>,
}

impl PartialOrd for SentenceScore {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl Eq for SentenceScore {}

impl Ord for SentenceScore {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.partial_cmp(&other.score).unwrap()
    }
}

pub fn batch_decode(
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch,
    s_batch: i32,
    output: &mut Vec<Vec<f32>>,
    normalise: bool,
) -> Result<()> {
    ctx.clear_kv_cache();
    ctx.decode(batch).with_context(|| "llama_decode() failed")?;

    for i in 0..s_batch {
        let embedding = ctx
            .embeddings_seq_ith(i)
            .with_context(|| "Failed to get embeddings")?;
        let output_embeddings = if normalise {
            normalize(embedding)
        } else {
            embedding.to_vec()
        };

        output.push(output_embeddings);
    }

    batch.clear();

    Ok(())
}

pub fn batch_decode_rerank(
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

pub fn single_decode(
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch,
    output: &mut Vec<f32>,
    normalise: bool,
) -> Result<()> {
    ctx.clear_kv_cache();
    ctx.decode(batch).with_context(|| "llama_decode() failed")?;
    let embedding = ctx
        .embeddings_seq_ith(0)
        .with_context(|| "Failed to get embeddings")?;
    let output_embeddings = if normalise {
        normalize(embedding)
    } else {
        embedding.to_vec()
    };
    output.extend(output_embeddings);
    batch.clear();
    Ok(())
}

pub fn get_embeddings(
    llama_tokens: &Vec<LlamaToken>,
    mut output: &mut Vec<Vec<f32>>,
    mut ctx: &mut LlamaContext,
    n_ctx: usize,
) -> Result<()> {
    let mut batch = LlamaBatch::new(n_ctx, 1);
    batch
        .add_sequence(&llama_tokens, 0, false)
        .with_context(|| "unable to add sequence to batch")?;

    batch_decode(&mut ctx, &mut batch, 1, &mut output, false)?;
    Ok(())
}

pub fn process_batch(
    ctx: &mut LlamaContext,
    splits_tokens: &Vec<Vec<LlamaToken>>,
) -> Result<Vec<Vec<f32>>, anyhow::Error> {
    let n_batch: usize = ctx.n_ctx() as usize;

    let mut batch = LlamaBatch::new(n_batch, splits_tokens.len() as i32);
    let mut max_seq_id_batch = 0;
    let mut output = Vec::with_capacity(splits_tokens.len());

    for tokens in splits_tokens {
        if batch.n_tokens() as usize + tokens.len() > n_batch {
            println!("Batch decode, n_tokens: {}, no_seq: {}", batch.n_tokens(), max_seq_id_batch);
            batch_decode(ctx, &mut batch, max_seq_id_batch, &mut output, true)?;
            max_seq_id_batch = 0;
        }
        batch.add_sequence(tokens, max_seq_id_batch, false)?;
        max_seq_id_batch += 1;
    }

    println!("Batch decode, n_tokens: {}, no_seq: {}", batch.n_tokens(), max_seq_id_batch);
    batch_decode(ctx, &mut batch, max_seq_id_batch, &mut output, true)?;

    Ok(output)
}

pub fn llama_cpp_tokenize(
    model: &LlamaModel,
    text: &str,
) -> Result<Vec<LlamaToken>, anyhow::Error> {
    let tokens = model.str_to_token(text, AddBos::Always)?;
    Ok(tokens)
}

pub fn hf_tokenize(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>, anyhow::Error> {
    let tokens = tokenizer
        .encode(text, true)
        .expect("failed to encode text")
        .get_ids()
        .to_vec();
    Ok(tokens)
}

pub fn hf_tokenize_fast(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>, anyhow::Error> {
    let tokens = tokenizer
        .encode_fast(text, true)
        .expect("failed to encode text")
        .get_ids()
        .to_vec();
    Ok(tokens)
}

pub fn process_splits_batch(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    splits: &Vec<String>,
) -> Result<Vec<Vec<f32>>, anyhow::Error> {
    let n_batch: usize = ctx.n_ctx() as usize;

    let mut batch = LlamaBatch::new(n_batch, splits.len() as i32);
    let mut max_seq_id_batch = 0;
    let mut output = Vec::with_capacity(splits.len());

    let splits_tokens = splits
        .iter()
        .map(|line| model.str_to_token(line.as_str(), AddBos::Always))
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("failed to tokenize {:?}", splits))?;

    for tokens in splits_tokens {
        if batch.n_tokens() as usize + tokens.len() > n_batch {
            //println!("Batch decode, n_tokens: {}, no_seq: {}", batch.n_tokens(), max_seq_id_batch);
            batch_decode(ctx, &mut batch, max_seq_id_batch, &mut output, true)?;
            max_seq_id_batch = 0;
        }
        batch.add_sequence(&tokens, max_seq_id_batch, false)?;
        max_seq_id_batch += 1;
    }

    batch_decode(ctx, &mut batch, max_seq_id_batch, &mut output, true)?;
    //println!("Batch decode, n_tokens: {}, no_seq: {}", batch.n_tokens(), max_seq_id_batch);

    Ok(output)
}

pub fn process_single(
    ctx: &mut LlamaContext,
    tokens: &Vec<LlamaToken>,
) -> Result<Vec<f32>, anyhow::Error> {
    let n_batch: usize = ctx.n_ctx() as usize;

    let mut batch = LlamaBatch::new(n_batch, 1);
    batch.add_sequence(tokens, 0, false)?;
    let mut output = Vec::with_capacity(1);
    single_decode(ctx, &mut batch, &mut output, true)?;
    Ok(output)
}

pub fn to_llama_tokens(hf_tokens: &Vec<u32>, model: &LlamaModel) -> Result<Vec<LlamaToken>> {
    let mut tokenized_chunk: Vec<_> = hf_tokens
        .iter()
        .map(|id| LlamaToken::new(*id as i32))
        .collect();
    tokenized_chunk.insert(0, model.token_bos());
    tokenized_chunk.push(model.token_eos());
    Ok(tokenized_chunk)
}

pub fn normalize(input: &[f32]) -> Vec<f32> {
    let magnitude = input
        .iter()
        .fold(0.0, |acc, &val| val.mul_add(val, acc))
        .sqrt();

    input.iter().map(|&val| val / magnitude).collect()
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

pub fn init_backend(log:bool) -> Result<LlamaBackend> {
    let mut backend = LlamaBackend::init()?;
    backend.void_logs();
    Ok(backend)
}

pub fn init_model(model_path: &str, backend: &LlamaBackend) -> Result<LlamaModel> {
    let model_params = if cfg!(any(feature = "cuda", feature = "metal")) {
        LlamaModelParams::default().with_n_gpu_layers(1000)
    } else {
        LlamaModelParams::default()
    };
    let model_path = PathBuf::from(model_path);
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)?;
    Ok(model)
}

pub fn init_context<'a>(
    model: &'a LlamaModel,
    backend: &'a LlamaBackend,
    max_tokens: Option<u32>,
    n_batch: Option<u32>,
    n_ubatch: Option<u32>,
) -> Result<LlamaContext<'a>> {
    let parallelism = std::thread::available_parallelism()?.get() as u32;
    println!("parallelism: {}", parallelism);
    let mut ctx_params = LlamaContextParams::default()
        //.with_n_threads(1)
        .with_n_threads_batch(parallelism.try_into()?)
        .with_embeddings(true);

    if let Some(max_tokens) = max_tokens {
        ctx_params = ctx_params
            .with_n_ctx(NonZeroU32::new(max_tokens))
            .with_n_ubatch(max_tokens);
    }

    if let Some(n_batch) = n_batch {
        ctx_params = ctx_params.with_n_batch(n_batch);
    }
    if let Some(n_ubatch) = n_ubatch {
        ctx_params = ctx_params.with_n_ubatch(n_ubatch);
    }

    ctx_params = ctx_params.with_pooling_type(LlamaPoolingType::Mean);


    let ctx = model.new_context(&backend, ctx_params)?;

    Ok(ctx)
}

pub fn init_reranker_context<'a>(
    model: &'a LlamaModel,
    backend: &'a LlamaBackend,
    pooling: Option<&str>,
    max_tokens: Option<u32>,
    n_batch: Option<u32>,
    n_ubatch: Option<u32>,
) -> Result<LlamaContext<'a>> {
    let pooling_type = match pooling {
        Some("mean") => LlamaPoolingType::Mean,
        Some("none") => LlamaPoolingType::None,
        Some("rank") => LlamaPoolingType::Rank,
        _ => LlamaPoolingType::Unspecified,
    };
    let parallelism = std::thread::available_parallelism()?.get() as u32;
    println!("parallelism: {}", parallelism);
    let mut ctx_params = LlamaContextParams::default()
        //.with_n_threads(1)
        .with_n_threads_batch(parallelism.try_into()?)
        .with_embeddings(true)
        .with_pooling_type(pooling_type);

    if let Some(max_tokens) = max_tokens {
        ctx_params = ctx_params
            .with_n_ctx(NonZeroU32::new(max_tokens))
            .with_n_ubatch(max_tokens);
    }

    if let Some(n_batch) = n_batch {
        ctx_params = ctx_params.with_n_batch(n_batch);
    }
    if let Some(n_ubatch) = n_ubatch {
        ctx_params = ctx_params.with_n_ubatch(n_ubatch);
    }

    ctx_params = ctx_params.with_pooling_type(LlamaPoolingType::Mean);


    let ctx = model.new_context(&backend, ctx_params)?;

    Ok(ctx)
}
pub fn init_splitter(
    model_id: Option<String>,
    patterns: Option<Vec<Vec<String>>>,
    max_tokens: Option<usize>,
    splits: bool,
) -> Result<SplitterLiteConfig<HFTokenizer>> {
    let patterns = patterns.unwrap_or(vec![
        vec!["\n\n".to_string()],
        vec!["\n".to_string()],
        vec![".".to_string(), "!".to_string(), "?".to_string()],
    ]);
    let max_tokens = max_tokens.unwrap_or(512);
    let merge_level = if splits { None } else { Some(patterns.len()) };
    let splitter_config = SplitterLiteConfig::new_hf(
        patterns.clone(),
        Some(max_tokens),
        merge_level,
        true,
        model_id.clone(),
    );
    Ok(splitter_config)
}
