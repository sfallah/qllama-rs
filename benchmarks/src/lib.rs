pub mod process_data;
pub mod split_data;

use anyhow::{anyhow, bail, Context, Result};
use fast_text_splitter::config::SplitterLiteConfig;
use fast_text_splitter::hf_tokenizer::HFTokenizer;
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use qllama::context::params::{LlamaContextParams, LlamaPoolingType};
use qllama::context::LlamaContext;
use qllama::llama_backend::LlamaBackend;
use qllama::llama_batch::LlamaBatch;
use qllama::model::params::LlamaModelParams;
use qllama::model::{AddBos, LlamaModel};
use qllama::token::LlamaToken;
use std::fmt::Debug;
use std::num::NonZeroU32;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Clone, PartialEq)]
pub struct SentenceScore {
    pub sentence: String,
    pub score: f64,
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
        output.push(if normalise {
            normalize(embedding)
        } else {
            embedding.to_vec()
        });
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
    pooling: &str,
) -> Result<()> {
    ctx.clear_kv_cache();
    ctx.decode(batch).with_context(|| "llama_decode() failed")?;

    for i in 0..s_batch {
        let embeddings = ctx
            .embeddings_seq_ith(i)
            .with_context(|| "Failed to get sequence embeddings")?;
        let normalized = if normalise {
            if pooling == "rank" {
                normalize_embeddings(embeddings, -1)
            } else {
                normalize_embeddings(embeddings, 2)
            }
        } else {
            embeddings.to_vec()
        };
        output.push(normalized);
    }

    batch.clear();
    Ok(())
}

/// Runs reranker decoding over tokenized prompts using context-sized batching.
pub fn rerank_token_batches(
    ctx: &mut LlamaContext,
    tokens_lines_list: &[Vec<LlamaToken>],
    max_tokens: usize,
    normalise: bool,
    pooling: &str,
) -> Result<Vec<Vec<f32>>> {
    let mut batch = LlamaBatch::new(max_tokens, 1);
    let mut max_seq_id_batch = 0;
    let mut output = Vec::with_capacity(tokens_lines_list.len());

    for tokens in tokens_lines_list {
        // Flush when the next sequence would exceed batch capacity.
        if (batch.n_tokens() as usize + tokens.len()) > max_tokens {
            batch_decode_rerank(
                ctx,
                &mut batch,
                max_seq_id_batch,
                &mut output,
                normalise,
                pooling,
            )?;
            max_seq_id_batch = 0;
            batch.clear();
        }
        batch.add_sequence(tokens, max_seq_id_batch, false)?;
        max_seq_id_batch += 1;
    }

    batch_decode_rerank(
        ctx,
        &mut batch,
        max_seq_id_batch,
        &mut output,
        normalise,
        pooling,
    )?;
    Ok(output)
}

/// Runs last-pooling reranker decoding over tokenized prompts using context-sized batching.
pub fn rerank_last_token_batches(
    ctx: &mut LlamaContext,
    tokens_lines_list: &[Vec<LlamaToken>],
    max_tokens: usize,
) -> Result<Vec<f32>> {
    let mut batch = LlamaBatch::new(max_tokens, 1);
    let mut max_seq_id_batch = 0;
    let mut output = Vec::with_capacity(tokens_lines_list.len());

    for tokens in tokens_lines_list {
        if (batch.n_tokens() as usize + tokens.len()) > max_tokens {
            batch_decode_rerank_last(ctx, &mut batch, &mut output, max_seq_id_batch)?;
            max_seq_id_batch = 0;
            batch.clear();
        }
        batch.add_sequence(tokens, max_seq_id_batch, false)?;
        max_seq_id_batch += 1;
    }

    batch_decode_rerank_last(ctx, &mut batch, &mut output, max_seq_id_batch)?;
    Ok(output)
}

/// Loads query + summary documents from a JSON file.
pub fn load_query_summaries(json_file_path: &str) -> Result<crate::split_data::QuerySummaries> {
    let input = std::fs::read_to_string(json_file_path)
        .with_context(|| format!("failed to read query summaries from `{json_file_path}`"))?;
    serde_json::from_str::<crate::split_data::QuerySummaries>(&input)
        .with_context(|| format!("failed to parse query summaries from `{json_file_path}`"))
}

/// Builds `<s>{query}</s></s>{document}</s>` style reranker prompts.
pub fn build_simple_reranker_prompts(query: &str, documents: &[String]) -> Vec<String> {
    documents
        .iter()
        .map(|doc| format!("<s>{query}</s></s>{doc}</s>"))
        .collect()
}

pub fn batch_decode_rerank_last(
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch,
    output: &mut Vec<f32>,
    s_batch: i32,
) -> Result<()> {
    ctx.clear_kv_cache();
    ctx.decode(batch).with_context(|| "llama_decode() failed")?;

    for i in 0..s_batch {
        let embed = ctx
            .embeddings_seq_ith(i)
            .with_context(|| "Failed to get sequence embeddings")?;
        output.push(embed[0].abs());
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
    output.extend(if normalise {
        normalize(embedding)
    } else {
        embedding.to_vec()
    });
    batch.clear();
    Ok(())
}

pub fn get_embeddings(
    llama_tokens: &[LlamaToken],
    output: &mut Vec<Vec<f32>>,
    ctx: &mut LlamaContext,
    n_ctx: usize,
) -> Result<()> {
    let mut batch = LlamaBatch::new(n_ctx, 1);
    batch
        .add_sequence(llama_tokens, 0, false)
        .with_context(|| "unable to add sequence to batch")?;
    batch_decode(ctx, &mut batch, 1, output, false)?;
    Ok(())
}

pub fn process_batch(
    ctx: &mut LlamaContext,
    splits_tokens: &[Vec<LlamaToken>],
) -> Result<Vec<Vec<f32>>> {
    let n_batch = ctx.n_ctx() as usize;
    let mut batch = LlamaBatch::new(n_batch, 1);
    let mut max_seq_id_batch = 0;
    let mut output = Vec::with_capacity(splits_tokens.len());

    for tokens in splits_tokens {
        if batch.n_tokens() as usize + tokens.len() > n_batch {
            batch_decode(ctx, &mut batch, max_seq_id_batch, &mut output, true)?;
            max_seq_id_batch = 0;
        }
        batch.add_sequence(tokens, max_seq_id_batch, false)?;
        max_seq_id_batch += 1;
    }

    batch_decode(ctx, &mut batch, max_seq_id_batch, &mut output, true)?;
    Ok(output)
}

pub fn llama_cpp_tokenize(model: &LlamaModel, text: &str) -> Result<Vec<LlamaToken>> {
    Ok(model.str_to_token(text, AddBos::Always)?)
}

pub fn hf_tokenize(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tokenizer
        .encode(text, true)
        .map_err(|e| anyhow!(e))?
        .get_ids()
        .to_vec())
}

pub fn hf_tokenize_fast(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tokenizer
        .encode_fast(text, true)
        .map_err(|e| anyhow!(e))?
        .get_ids()
        .to_vec())
}

pub fn process_splits_batch(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    splits: &[String],
) -> Result<Vec<Vec<f32>>> {
    let n_batch = ctx.n_ctx() as usize;
    let mut batch = LlamaBatch::new(n_batch, 1);
    let mut max_seq_id_batch = 0;
    let mut output = Vec::with_capacity(splits.len());

    let splits_tokens = splits
        .iter()
        .map(|line| model.str_to_token(line.as_str(), AddBos::Always))
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("failed to tokenize {:?}", splits))?;

    for tokens in splits_tokens {
        if batch.n_tokens() as usize + tokens.len() > n_batch {
            batch_decode(ctx, &mut batch, max_seq_id_batch, &mut output, true)?;
            max_seq_id_batch = 0;
        }
        batch.add_sequence(&tokens, max_seq_id_batch, true)?;
        max_seq_id_batch += 1;
    }

    batch_decode(ctx, &mut batch, max_seq_id_batch, &mut output, true)?;
    Ok(output)
}

pub fn process_single(ctx: &mut LlamaContext, tokens: &[LlamaToken]) -> Result<Vec<f32>> {
    let n_batch = ctx.n_ctx() as usize;
    let mut batch = LlamaBatch::new(n_batch, 0);
    batch.add_sequence(tokens, 0, false)?;
    let mut output = Vec::with_capacity(1);
    single_decode(ctx, &mut batch, &mut output, true)?;
    Ok(output)
}

pub fn to_llama_tokens(hf_tokens: &[u32], model: &LlamaModel) -> Result<Vec<LlamaToken>> {
    let tokens: Vec<LlamaToken> = std::iter::once(model.token_bos())
        .chain(hf_tokens.iter().map(|&id| LlamaToken::new(id as i32)))
        .chain(std::iter::once(model.token_eos()))
        .collect();
    Ok(tokens)
}

pub fn normalize(input: &[f32]) -> Vec<f32> {
    let magnitude = input
        .iter()
        .fold(0.0f32, |acc, &val| val.mul_add(val, acc))
        .sqrt();
    input.iter().map(|&val| val / magnitude).collect()
}

fn normalize_embeddings(input: &[f32], embd_norm: i32) -> Vec<f32> {
    let sum: f64 = match embd_norm {
        -1 => 1.0,
        0 => (input.iter().map(|x| x.abs()).fold(0.0f32, f32::max) / 32760.0) as f64,
        2 => input
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            .sqrt(),
        p => input
            .iter()
            .map(|x| (x.abs() as f64).powi(p))
            .sum::<f64>()
            .powf(1.0 / p as f64),
    };

    let norm = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    input.iter().map(|&x| (x as f64 * norm) as f32).collect()
}

pub fn init_backend(log: bool) -> Result<LlamaBackend> {
    let mut backend = LlamaBackend::init()?;
    if !log {
        backend.void_logs();
    }
    Ok(backend)
}

fn make_model_params() -> LlamaModelParams {
    if cfg!(any(feature = "cuda", feature = "metal")) {
        LlamaModelParams::default().with_n_gpu_layers(1000)
    } else {
        LlamaModelParams::default()
    }
}

pub fn init_model(model_path: &str, backend: &LlamaBackend) -> Result<LlamaModel> {
    Ok(LlamaModel::load_from_file(
        backend,
        PathBuf::from(model_path),
        &make_model_params(),
    )?)
}

/// Returns the local cached path (as a String) for a file in a Hugging Face model repository,
/// downloading it first if needed.
///
/// When `revision` is `None`, the repository's default `main` revision is used.
///
/// ```no_run
/// use qllama_bench::ensure_hf_model_file;
///
/// let model_path = ensure_hf_model_file(
///     "BAAI/bge-m3",
///     "bge-m3-q4_k_m.gguf",
///     None,
/// ).unwrap();
/// ```
pub fn ensure_hf_model_file(
    repo_id: &str,
    filename: &str,
    revision: Option<&str>,
) -> Result<String> {
    let repo_id = repo_id.trim();
    if repo_id.is_empty() {
        bail!("Hugging Face repo id cannot be empty");
    }

    let filename = filename.trim();
    if filename.is_empty() {
        bail!("Hugging Face filename cannot be empty");
    }

    let revision = revision.map(str::trim).filter(|r| !r.is_empty());

    let api = Api::new().with_context(|| "Failed to create Hugging Face Hub API client")?;
    let repo = match revision {
        Some(rev) => api.repo(Repo::with_revision(
            repo_id.to_string(),
            RepoType::Model,
            rev.to_string(),
        )),
        None => api.model(repo_id.to_string()),
    };

    let path_buf = repo.get(filename).with_context(|| match revision {
        Some(rev) => format!("Failed to fetch `{filename}` from `{repo_id}` at revision `{rev}`"),
        None => format!("Failed to fetch `{filename}` from `{repo_id}`"),
    })?;

    path_buf
        .to_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("Model path contains invalid UTF-8"))
}

pub fn init_model_multi(
    model_path: &str,
    backend: &LlamaBackend,
    n_models: usize,
) -> Vec<Result<LlamaModel>> {
    let path = PathBuf::from(model_path);
    (0..n_models)
        .map(|_| {
            LlamaModel::load_from_file(backend, path.clone(), &make_model_params())
                .map_err(|e| anyhow!("Failed to load model: {}", e))
        })
        .collect()
}

pub fn init_context<'a>(
    model: &'a LlamaModel,
    backend: &'a LlamaBackend,
    max_tokens: Option<u32>,
    n_batch: Option<u32>,
    n_ubatch: Option<u32>,
) -> Result<LlamaContext<'a>> {
    let parallelism = std::thread::available_parallelism()?.get() as u32;
    let mut ctx_params = LlamaContextParams::default()
        .with_n_threads_batch(parallelism.try_into()?)
        .with_embeddings(true)
        .with_kv_unified(true)
        .with_pooling_type(LlamaPoolingType::Mean);

    if let Some(t) = max_tokens {
        ctx_params = ctx_params.with_n_ctx(NonZeroU32::new(t)).with_n_ubatch(t);
    }
    if let Some(b) = n_batch {
        ctx_params = ctx_params.with_n_batch(b);
    }
    if let Some(ub) = n_ubatch {
        ctx_params = ctx_params.with_n_ubatch(ub);
    }

    Ok(model.new_context(backend, ctx_params)?)
}

pub fn init_reranker_context<'a>(
    model: &'a LlamaModel,
    backend: &'a LlamaBackend,
    max_tokens: u32,
    pooling: Option<LlamaPoolingType>,
) -> Result<LlamaContext<'a>> {
    let parallelism = std::thread::available_parallelism()?.get() as u32;
    let ctx_params = LlamaContextParams::default()
        .with_n_threads_batch(parallelism.try_into()?)
        .with_embeddings(true)
        .with_pooling_type(pooling.unwrap_or(LlamaPoolingType::Rank))
        .with_n_ctx(NonZeroU32::new(max_tokens))
        .with_n_ubatch(max_tokens)
        .with_kv_unified(true)
        .with_n_batch(max_tokens);
    Ok(model.new_context(backend, ctx_params)?)
}

pub fn init_splitter(
    model_id: Option<String>,
    patterns: Option<Vec<Vec<String>>>,
    max_tokens: Option<usize>,
    splits: bool,
) -> Result<SplitterLiteConfig<HFTokenizer>> {
    let patterns = patterns.unwrap_or_else(|| {
        vec![
            vec!["<SENT>".to_string()],
            vec!["\n\n".to_string()],
            vec!["\n".to_string()],
        ]
    });
    let max_tokens = max_tokens.unwrap_or(512);
    let merge_level = if splits { None } else { Some(patterns.len()) };
    Ok(SplitterLiteConfig::new_hf(
        patterns,
        Some(max_tokens),
        merge_level,
        true,
        model_id,
    ))
}

/// Computes the probability for the "yes" token given logits for a single example.
///
/// # Arguments
/// * `logits` - Slice of logits for the vocabulary (length = vocab size)
/// * `token_true_id` - Index of the "yes" token in the vocabulary
/// * `token_false_id` - Index of the "no" token in the vocabulary
///
/// # Returns
/// Probability for the "yes" token (as f32)
pub fn compute_logits(logits: &[f32], token_true_id: usize, token_false_id: usize) -> f32 {
    let yes_logit = logits[token_true_id];
    let no_logit = logits[token_false_id];
    let scores = [no_logit, yes_logit];
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = scores.iter().map(|&x| (x - max_score).exp()).sum();
    let yes_log_softmax = yes_logit - (max_score + exp_sum.ln());
    yes_log_softmax.exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensure_hf_model_file_rejects_empty_repo_id() {
        let err = ensure_hf_model_file("   ", "config.json", None).unwrap_err();
        assert!(err.to_string().contains("repo id cannot be empty"));
    }

    #[test]
    fn test_ensure_hf_model_file_rejects_empty_filename() {
        let err = ensure_hf_model_file("bert-base-uncased", "   ", None).unwrap_err();
        assert!(err.to_string().contains("filename cannot be empty"));
    }

    #[test]
    fn test_compute_logits_basic() {
        let logits = vec![0.0, 1.0, 2.0, 3.0];
        let prob_yes = compute_logits(&logits, 2, 1);
        let expected =
            (2.0f32 - (2.0f32.max(1.0) + ((2.0f32 - 2.0).exp() + (1.0f32 - 2.0).exp()).ln())).exp();
        assert!((prob_yes - expected).abs() < 1e-6);
    }
}
