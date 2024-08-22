use anyhow::{Context, Result};
use fast_text_splitter::config::SplitterLiteConfig;
use fast_text_splitter::hf_tokenizer::HFTokenizer;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use std::fmt::Debug;

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


pub fn get_splitter_config(model_id: Option<String>, max_tokens: Option<usize>) -> Result<SplitterLiteConfig<HFTokenizer>> {
    let patterns = vec![
        vec!["\n\n".to_string()],
        vec!["\n".to_string()],
        vec![".".to_string(), "!".to_string(), "?".to_string()],
    ];
    let splitter_config =
        SplitterLiteConfig::new_hf(patterns.clone(), max_tokens, None, true, model_id);
    Ok(splitter_config)
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
            //println!("Batch decode, n_tokens: {}, no_seq: {}", batch.n_tokens(), max_seq_id_batch);
            batch_decode(ctx, &mut batch, max_seq_id_batch, &mut output, true)?;
            max_seq_id_batch = 0;
        }
        batch.add_sequence(tokens, max_seq_id_batch, false)?;
        max_seq_id_batch += 1;
    }

    batch_decode(ctx, &mut batch, max_seq_id_batch, &mut output, true)?;
    //println!("Batch decode, n_tokens: {}, no_seq: {}", batch.n_tokens(), max_seq_id_batch);

    Ok(output)
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

    let splits_tokens = splits.iter()
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