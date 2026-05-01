pub mod apply_template;
pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod health;
pub mod infill;
pub mod lora;
pub mod metrics;
pub mod models;
pub mod props;
pub mod rerank;
pub mod router;
pub mod slots;
pub mod tokenize;
pub mod transcriptions;

use crate::engine::task::{TaskKind, TaskResult};
use crate::http::error::{AppError, AppResult};
use crate::state::AppState;
use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use serde_json::{json, Value};

pub async fn submit_and_wait(state: &AppState, kind: TaskKind, payload: Value) -> AppResult<Value> {
    let handle = state
        .engine
        .submit(kind, payload)
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    let mut rx = handle.result_rx.resubscribe();
    loop {
        match rx.recv().await {
            Ok(TaskResult::Done(v)) => return Ok(v),
            Ok(TaskResult::Error(e)) => return Err(AppError::Internal(e)),
            Ok(TaskResult::Chunk(_)) => continue,
            Err(e) => return Err(AppError::Internal(e.to_string())),
        }
    }
}

pub async fn generic_task_handler(
    State(state): State<AppState>,
    kind: TaskKind,
    payload: Value,
) -> impl IntoResponse {
    match submit_and_wait(&state, kind, payload).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => e.into_response(),
    }
}

pub fn not_implemented_response(name: &str) -> Value {
    json!({
        "status": "not_implemented",
        "handler": name,
    })
}
