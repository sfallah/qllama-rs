#![allow(missing_docs)]

mod cli;
mod config;
mod engine;
mod http;
mod state;

use crate::cli::Cli;
use crate::config::ServerConfig;
use crate::engine::Engine;
use crate::http::build_router;
use crate::state::AppState;
use clap::Parser;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    let config = Arc::new(ServerConfig::from_cli(cli)?);
    let engine = Engine::new(Arc::clone(&config));
    let state = AppState::new(Arc::clone(&config), engine);

    let app = build_router(state);
    let addr = format!("{}:{}", config.host, config.port);

    if let (Some(cert), Some(key)) = (config.tls_cert.as_deref(), config.tls_key.as_deref()) {
        let tls_config = axum_server::tls_rustls::RustlsConfig::from_pem_file(cert, key).await?;
        println!("llama-server-rs listening on https://{addr}");
        axum_server::bind_rustls(addr.parse()?, tls_config)
            .serve(app.into_make_service())
            .await?;
        return Ok(());
    }

    println!("llama-server-rs listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}

#[allow(dead_code)]
fn _fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../llama-cpp/src/gguf/ggml-vocab-bert-bge.gguf")
}
