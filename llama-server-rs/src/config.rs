use crate::cli::{Cli, ModelArg};
use anyhow::Context;
use hf_hub::api::sync::ApiBuilder;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub tls_cert: Option<PathBuf>,
    pub tls_key: Option<PathBuf>,
    pub api_keys: Vec<String>,
    pub model_path: PathBuf,
    pub model_id: String,
    pub slots: usize,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_predict: i32,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
}

impl ServerConfig {
    pub fn from_cli(cli: Cli) -> anyhow::Result<Self> {
        let model_path = resolve_model(cli.model)?;
        let model_id = model_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("llama.cpp")
            .to_string();

        Ok(Self {
            host: cli.host,
            port: cli.port,
            tls_cert: cli.tls_cert,
            tls_key: cli.tls_key,
            api_keys: cli.api_keys,
            model_path,
            model_id,
            slots: cli.slots.max(1),
            n_ctx: cli.n_ctx,
            n_batch: cli.n_batch,
            n_ubatch: cli.n_ubatch,
            n_predict: cli.n_predict,
            temperature: cli.temperature,
            top_k: cli.top_k,
            top_p: cli.top_p,
            min_p: cli.min_p,
        })
    }
}

fn resolve_model(model: ModelArg) -> anyhow::Result<PathBuf> {
    match model {
        ModelArg::Local { path } => Ok(path),
        ModelArg::HuggingFace { repo, model } => ApiBuilder::new()
            .with_progress(true)
            .build()
            .context("failed to initialize HF API")?
            .model(repo)
            .get(&model)
            .context("failed to download model from HF"),
    }
}
