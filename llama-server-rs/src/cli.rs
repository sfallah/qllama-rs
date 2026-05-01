use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(author, version, about = "llama.cpp REST API server (Axum)")]
pub struct Cli {
    #[command(subcommand)]
    pub model: ModelArg,

    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    #[arg(long)]
    pub tls_cert: Option<PathBuf>,

    #[arg(long)]
    pub tls_key: Option<PathBuf>,

    #[arg(long = "api-key")]
    pub api_keys: Vec<String>,

    #[arg(long, default_value_t = 1)]
    pub slots: usize,

    #[arg(long = "ctx-size", default_value_t = 4096)]
    pub n_ctx: u32,

    #[arg(long = "batch-size", default_value_t = 512)]
    pub n_batch: u32,

    #[arg(long = "ubatch-size", default_value_t = 512)]
    pub n_ubatch: u32,

    #[arg(long = "predict", default_value_t = 256)]
    pub n_predict: i32,

    #[arg(long, default_value_t = 0.8)]
    pub temperature: f32,

    #[arg(long, default_value_t = 40)]
    pub top_k: i32,

    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,

    #[arg(long, default_value_t = 0.05)]
    pub min_p: f32,
}

#[derive(Debug, Subcommand)]
pub enum ModelArg {
    Local {
        path: PathBuf,
    },
    #[command(name = "hf-model")]
    HuggingFace {
        repo: String,
        model: String,
    },
}
