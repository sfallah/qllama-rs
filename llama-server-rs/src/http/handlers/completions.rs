use crate::engine::task::TaskKind;
use crate::http::handlers::submit_and_wait;
use crate::state::AppState;
use axum::extract::State;
use axum::Json;
use serde_json::Value;

pub async fn post_completion(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let result = submit_and_wait(&state, TaskKind::Completion, payload)
        .await
        .unwrap_or_else(|e| serde_json::json!({"error": e.to_string()}));
    Json(result)
}

pub async fn post_completions_oai(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let result = submit_and_wait(&state, TaskKind::CompletionsOai, payload)
        .await
        .unwrap_or_else(|e| serde_json::json!({"error": e.to_string()}));
    Json(result)
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
