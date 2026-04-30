use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use llama_cpp_rs_bench::ensure_hf_model_file;
use llama_cpp_rs_bench::process_data::{json_file_embeddings, text_file_embeddings};

fn optional_str_value(value: &str) -> Option<&str> {
    if value == "-" {
        None
    } else {
        Some(value)
    }
}

fn optional_owned_value(value: &str) -> Option<String> {
    if value == "-" {
        None
    } else {
        Some(value.to_string())
    }
}

fn parse_optional_u32(value: &str) -> Result<Option<u32>> {
    if value == "-" {
        Ok(None)
    } else {
        Ok(Some(value.parse()?))
    }
}

#[derive(Parser, Debug)]
#[command(name = "process-data")]
#[command(about = "Generate embedding output files from text/json inputs", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Text(TextArgs),
    Json(JsonArgs),
}

#[derive(Args, Debug)]
struct TextArgs {
    /// Hugging Face repo id for the model file
    #[arg(
        long,
        default_value = "sabafallah/embeddinggemma-300m-sentence-transformers-gguf"
    )]
    repo_id: String,
    /// Model filename inside the Hugging Face repo
    #[arg(
        long,
        default_value = "embeddinggemma-300m-sentence-transformers-q8_0.gguf"
    )]
    filename: String,
    #[arg(long, default_value = "tests/test_data/superlinear.txt")]
    text_file_path: String,
    #[arg(long, default_value = "output/superlinear_embeddings/embeddinggemma")]
    out_dir: String,
    /// HuggingFace tokenizer model id, or '-' for None
    #[arg(long, default_value = "google/embeddinggemma-300m")]
    hf_model: String,
    /// Optional instruction prefix, or '-' for None
    #[arg(long, default_value = "task: sentence similarity | query: ")]
    model_instruct: String,
    /// Optional n_ctx, or '-' for None
    #[arg(long, default_value = "512")]
    n_ctx: String,
    /// Optional n_batch, or '-' for None
    #[arg(long, default_value = "512")]
    n_batch: String,
    /// Optional n_ubatch, or '-' for None
    #[arg(long, default_value = "512")]
    n_ubatch: String,
}

#[derive(Args, Debug)]
struct JsonArgs {
    /// Hugging Face repo id for the model file
    #[arg(long, default_value = "sabafallah/bge-m3-Q4_K_M-GGUF")]
    repo_id: String,
    /// Model filename inside the Hugging Face repo
    #[arg(long, default_value = "bge-m3-q4_k_m.gguf")]
    filename: String,
    #[arg(long, default_value = "tests/test_data/gold_extractive.json")]
    json_file_path: String,
    #[arg(long, default_value = "output/gold_extractive/bge-m3")]
    out_dir: String,
    /// HuggingFace tokenizer model id, or '-' for None
    #[arg(long, default_value = "BAAI/bge-m3")]
    hf_model: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Text(args) => {
            let model_path = ensure_hf_model_file(&args.repo_id, &args.filename, None)?;
            let model_instruct = optional_str_value(&args.model_instruct);
            let n_ctx = parse_optional_u32(&args.n_ctx)?;
            let n_batch = parse_optional_u32(&args.n_batch)?;
            let n_ubatch = parse_optional_u32(&args.n_ubatch)?;

            text_file_embeddings(
                &model_path,
                &args.text_file_path,
                &args.out_dir,
                optional_owned_value(&args.hf_model),
                model_instruct,
                n_ctx,
                n_batch,
                n_ubatch,
            )?;
        }
        Commands::Json(args) => {
            let model_path = ensure_hf_model_file(&args.repo_id, &args.filename, None)?;
            json_file_embeddings(
                &model_path,
                &args.json_file_path,
                &args.out_dir,
                optional_owned_value(&args.hf_model),
            )?;
        }
    }

    Ok(())
}
