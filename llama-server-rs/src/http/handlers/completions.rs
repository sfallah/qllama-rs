use crate::engine::task::{TaskKind, TaskResult};
use crate::http::error::{AppError, AppResult};
use crate::http::handlers::submit_and_wait;
use crate::http::sse::{stream_completion_response, StreamFraming};
use crate::state::AppState;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::Value;

pub async fn post_completion(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> AppResult<Response> {
    run_completion(state, payload, TaskKind::Completion, StreamFraming::LegacySse).await
}

pub async fn post_completions_oai(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> AppResult<Response> {
    run_completion(
        state,
        payload,
        TaskKind::CompletionsOai,
        StreamFraming::OaiSse,
    )
    .await
}

async fn run_completion(
    state: AppState,
    payload: Value,
    kind: TaskKind,
    framing: StreamFraming,
) -> AppResult<Response> {
    if !state.engine.is_ready() {
        return Err(AppError::Unavailable("Loading model".to_string()));
    }

    let stream = payload.get("stream").and_then(Value::as_bool).unwrap_or(false);
    if !stream {
        return Ok(Json(submit_and_wait(&state, kind, payload).await?).into_response());
    }

    let mut handle = state
        .engine
        .submit(kind, payload)
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    // parity with the reference server: a failure on the first result is reported
    // as a plain JSON error instead of opening an event stream.
    match handle.recv().await {
        Some(TaskResult::Error(err)) => Err(AppError::from(err)),
        Some(first) => Ok(stream_completion_response(first, handle, framing)),
        None => Err(AppError::Internal("result channel closed".to_string())),
    }
}

pub async fn post_responses(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let result = submit_and_wait(&state, TaskKind::Responses, payload)
        .await
        .unwrap_or_else(|e| serde_json::json!({"error": e.to_string()}));
    Json(result)
}

pub async fn post_anthropic_messages(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let result = submit_and_wait(&state, TaskKind::AnthropicMessages, payload)
        .await
        .unwrap_or_else(|e| serde_json::json!({"error": e.to_string()}));
    Json(result)
}

pub async fn post_anthropic_count_tokens(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let result = submit_and_wait(&state, TaskKind::AnthropicCountTokens, payload)
        .await
        .unwrap_or_else(|e| serde_json::json!({"error": e.to_string()}));
    Json(result)
}
