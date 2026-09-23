use anyhow::Result;
use candle_core::{Device, Tensor};
use fast_text_splitter::config::SplitterLiteConfig;
use fast_text_splitter::hf_tokenizer::HFTokenizer;
use fast_text_splitter::splitter::split_node::utils::SplitResultLite;
use qllama::context::LlamaContext;
use qllama::token::LlamaToken;
use std::fs;
use std::path::Path;

use crate::split_data::{SplitData, SummaryData};
use crate::{
    get_embeddings, init_backend, init_context, init_model, init_splitter, llama_cpp_tokenize,
    process_batch,
};

fn ensure_dir_exists(dir_path: &str) -> std::io::Result<()> {
    if !Path::new(dir_path).exists() {
        fs::create_dir_all(dir_path)?;
    }
    Ok(())
}

fn process_split_data(
    out_dir: &str,
    ctx: &mut LlamaContext,
    n_ctx: usize,
    sentence_splitter: &SplitterLiteConfig<HFTokenizer>,
    device: &Device,
    splits_data: &mut Vec<SplitData>,
    split_id: usize,
    mx_tokens_split: &SplitResultLite,
    model_instruct: Option<&str>,
) {
    let split_id_str = format!("{:03}", split_id);

    let input = model_instruct.map_or_else(
        || mx_tokens_split.split_string.clone(),
        |instruct| format!("{}{}", instruct, mx_tokens_split.split_string),
    );

    let llama_tokens = llama_cpp_tokenize(&ctx.model, input.as_str()).expect("unable to tokenize");
    let mut split_embedding_vec = Vec::new();
    get_embeddings(&llama_tokens, &mut split_embedding_vec, ctx, n_ctx).expect("embeddings failed");

    let split_embedding_tensor =
        Tensor::new(split_embedding_vec, device).expect("unable to create tensor");
    let split_embedding_name = format!("split_embedding_{}", split_id_str);
    let split_embedding_file = format!("{}.safetensors", split_embedding_name);
    let split_embedding_path = format!("{}/{}", out_dir, split_embedding_file);

    split_embedding_tensor
        .save_safetensors(split_embedding_name.as_str(), split_embedding_path.as_str())
        .expect("unable to save tensors");

    let sentence_splits = sentence_splitter.hf_splits(mx_tokens_split.split_string.as_bytes());

    let sentence_splits_strs = sentence_splits
        .iter()
        .filter(|sentence_split| !sentence_split.tokens.is_empty())
        .map(|sentence_split| sentence_split.split_string.clone())
        .collect::<Vec<String>>();

    let llama_tokens_list: Vec<Vec<LlamaToken>> = sentence_splits_strs
        .iter()
        .map(|sentence_str| {
            llama_cpp_tokenize(&ctx.model, sentence_str).expect("unable to convert to llama tokens")
        })
        .collect();
    let embds = process_batch(ctx, &llama_tokens_list).expect("unable to process batch");
    let tensors = Tensor::new(embds, device).expect("unable to create tensor");

    let sentence_embeddings_name = format!("sentence_embeddings_{}", split_id_str);
    let sentence_embeddings_file = format!("{}.safetensors", sentence_embeddings_name);
    let sentence_embeddings_path = format!("{}/{}", out_dir, sentence_embeddings_file);

    let split_data = SplitData {
        split_id,
        no_tokens: mx_tokens_split.tokens.len(),
        split_embedding_name: split_embedding_name.clone(),
        split_embedding_file: split_embedding_file.clone(),
        split_string: mx_tokens_split.split_string.clone(),
        sentence_embeddings_name: sentence_embeddings_name.clone(),
        sentence_embeddings_file: sentence_embeddings_file.clone(),
        sentences: sentence_splits_strs.clone(),
    };
    splits_data.push(split_data);

    tensors
        .save_safetensors(
            sentence_embeddings_name.as_str(),
            sentence_embeddings_path.as_str(),
        )
        .expect("unable to save tensors");
}

fn process_inputs_embeddings(
    out_dir: &str,
    ctx: &mut LlamaContext,
    n_ctx: usize,
    split_splitter: &SplitterLiteConfig<HFTokenizer>,
    sentence_splitter: &SplitterLiteConfig<HFTokenizer>,
    device: &Device,
    inputs: Vec<String>,
    model_instruct: Option<&str>,
) -> Result<()> {
    let mx_tokens_splits = inputs
        .into_iter()
        .flat_map(|input| split_splitter.hf_splits(input.as_bytes()))
        .collect::<Vec<SplitResultLite>>();

    let mut splits_data = Vec::new();
    mx_tokens_splits
        .iter()
        .enumerate()
        .for_each(|(split_id, mx_tokens_split)| {
            process_split_data(
                out_dir,
                ctx,
                n_ctx,
                sentence_splitter,
                device,
                &mut splits_data,
                split_id,
                mx_tokens_split,
                model_instruct,
            );
        });

    let splits_data_json = serde_json::to_string_pretty(&splits_data)?;
    let splits_data_file = format!("{}/splits_data.json", out_dir);
    fs::write(splits_data_file, splits_data_json)?;

    Ok(())
}

/// Generates split-level and sentence-level embedding artifacts from a plain text file.
///
/// This is a high-level data-processing entrypoint intended for offline dataset generation.
/// The function performs the full pipeline below:
///
/// 1. Initializes backend/model/context (`init_backend`, `init_model`, `init_context`).
/// 2. Builds two splitters using the provided `hf_model` tokenizer id:
///    - a **coarse split splitter** (`splits = true`) used to partition the input text
///      into main chunks,
///    - a **sentence splitter** (`splits = false`) used per coarse chunk.
/// 3. Reads `text_file_path` into memory as UTF-8 text.
/// 4. For each coarse split:
///    - optionally prefixes text with `model_instruct` before tokenization,
///    - computes one split embedding tensor,
///    - computes per-sentence embeddings for sentence sub-splits,
///    - writes tensors as `.safetensors` files under `out_dir`.
/// 5. Writes a `splits_data.json` manifest in `out_dir` describing all generated artifacts.
///
/// # Output layout
///
/// The function writes files into `out_dir` (creating it if missing), including:
///
/// - `split_embedding_XXX.safetensors` for each coarse split,
/// - `sentence_embeddings_XXX.safetensors` for sentence embeddings of that split,
/// - `splits_data.json` metadata linking split text and generated file names.
///
/// `XXX` is a zero-padded split index (`000`, `001`, ...).
///
/// # Parameters
///
/// - `model_path`: local model file path (typically a GGUF file).
/// - `text_file_path`: source text file to process.
/// - `out_dir`: destination directory for generated safetensors + manifest.
/// - `hf_model`: optional Hugging Face tokenizer id used by splitters.
/// - `model_instruct`: optional prefix prepended to each coarse split before model tokenization.
///   This changes embeddings by design and is useful for instruction-tuned embedding models.
/// - `n_ctx`, `n_batch`, `n_ubatch`: optional context/batching overrides forwarded to
///   `init_context`; use `None` to keep library defaults.
///
/// # Errors
///
/// Returns an error if backend/model/context setup fails, file I/O fails, tokenizer/splitting
/// fails, embedding extraction fails, or any output artifact cannot be written.
///
/// # Example
///
/// ```no_run
/// use qllama_bench::process_data::text_file_embeddings;
///
/// text_file_embeddings(
///     "models/my-embedding-model.gguf",
///     "tests/test_data/superlinear.txt",
///     "output/superlinear_embeddings/my-model",
///     Some("google/embeddinggemma-300m".to_string()),
///     None,
///     Some(512),
///     Some(512),
///     Some(512),
/// )?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn text_file_embeddings(
    model_path: &str,
    text_file_path: &str,
    out_dir: &str,
    hf_model: Option<String>,
    model_instruct: Option<&str>,
    n_ctx: Option<u32>,
    n_batch: Option<u32>,
    n_ubatch: Option<u32>,
) -> Result<()> {
    let backend = init_backend(false)?;
    let model = init_model(model_path, &backend)?;
    let mut ctx = init_context(&model, &backend, n_ctx, n_batch, n_ubatch)?;

    let n_ctx = ctx.n_ctx() as usize;

    let split_splitter = init_splitter(hf_model.clone(), None, Some(400), true)?;
    let sentence_splitter = init_splitter(hf_model, None, Some(400), false)?;

    let device = Device::Cpu;
    ensure_dir_exists(out_dir)?;

    let binding = fs::read_to_string(text_file_path)?;
    process_inputs_embeddings(
        out_dir,
        &mut ctx,
        n_ctx,
        &split_splitter,
        &sentence_splitter,
        &device,
        vec![binding],
        model_instruct,
    )?;

    Ok(())
}

/// Generates split-level and sentence-level embedding artifacts from a JSON file containing
/// `text` / `summary` pairs.
///
/// This is a high-level data-processing entrypoint intended for offline dataset generation.
/// The JSON file is expected to be an array of [`SummaryData`] objects, each carrying a `text`
/// field and a `summary` field.  The function runs the full embedding pipeline **twice** — once
/// for the `text` fields and once for the `summary` fields — and writes the results into two
/// separate sub-directories of `out_dir`.
///
/// # Pipeline
///
/// 1. Initializes backend / model / context (`init_backend`, `init_model`, `init_context` with
///    `n_ctx = 4096`).
/// 2. Builds two splitters using the provided `hf_model` tokenizer id:
///    - a **coarse split splitter** (`splits = true`, max tokens 4096),
///    - a **sentence splitter** (`splits = false`, max tokens 4096).
/// 3. Deserialises `json_file_path` into a `Vec<SummaryData>`.
/// 4. Runs [`process_inputs_embeddings`] on all `text` fields → writes into `{out_dir}/text/`.
/// 5. Runs [`process_inputs_embeddings`] on all `summary` fields → writes into
///    `{out_dir}/summary/`.
///
/// Each pass produces, per coarse split:
/// - `split_embedding_XXX.safetensors` — one embedding for the whole coarse chunk,
/// - `sentence_embeddings_XXX.safetensors` — stacked per-sentence embeddings,
/// - `splits_data.json` — manifest linking split text and generated file names.
///
/// `XXX` is a zero-padded global split index (`000`, `001`, …) across all input documents
/// within a pass.
///
/// # Parameters
///
/// - `model_path`: local model file path (typically a GGUF file).
/// - `json_file_path`: path to a JSON file deserializable as `Vec<SummaryData>`.
/// - `out_dir`: root destination directory; `text/` and `summary/` sub-directories are
///   created automatically if they do not exist.
/// - `hf_model`: optional Hugging Face tokenizer id used by both splitters.
///
/// # Errors
///
/// Returns an error if backend / model / context setup fails, file I/O fails,
/// JSON deserialisation fails, tokenizer / splitting fails, embedding extraction fails,
/// or any output artifact cannot be written.
///
/// # Example
///
/// ```no_run
/// use qllama_bench::process_data::json_file_embeddings;
///
/// json_file_embeddings(
///     "models/bge-m3-q4_k_m.gguf",
///     "tests/test_data/gold_extractive.json",
///     "output/gold_extractive/bge-m3",
///     Some("BAAI/bge-m3".to_string()),
/// )?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn json_file_embeddings(
    model_path: &str,
    json_file_path: &str,
    out_dir: &str,
    hf_model: Option<String>,
) -> Result<()> {
    let backend = init_backend(true)?;
    let model = init_model(model_path, &backend)?;
    let mut ctx = init_context(&model, &backend, Some(4096), None, None)?;

    let n_ctx = ctx.n_ctx() as usize;

    let split_splitter = init_splitter(hf_model.clone(), None, Some(4096), true)?;
    let sentence_splitter = init_splitter(hf_model, None, Some(4096), false)?;

    let device = Device::Cpu;

    let text_out_dir = format!("{}/text", out_dir);
    ensure_dir_exists(text_out_dir.as_str())?;

    let input_str = fs::read_to_string(json_file_path)?;
    let inputs_vec: Vec<SummaryData> = serde_json::from_str(&input_str)?;

    process_inputs_embeddings(
        text_out_dir.as_str(),
        &mut ctx,
        n_ctx,
        &split_splitter,
        &sentence_splitter,
        &device,
        inputs_vec.iter().map(|input| input.text.clone()).collect(),
        None,
    )?;

    let summary_out_dir = format!("{}/summary", out_dir);
    ensure_dir_exists(summary_out_dir.as_str())?;

    process_inputs_embeddings(
        summary_out_dir.as_str(),
        &mut ctx,
        n_ctx,
        &split_splitter,
        &sentence_splitter,
        &device,
        inputs_vec.into_iter().map(|input| input.summary).collect(),
        None,
    )?;

    Ok(())
}
