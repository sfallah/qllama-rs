use crate::engine::task::TaskKind;
use crate::http::handlers::submit_and_wait;
use crate::state::AppState;
use axum::extract::State;
use axum::Json;
use serde_json::Value;

pub async fn post_chat_completions(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let result = submit_and_wait(&state, TaskKind::ChatCompletions, payload)
        .await
        .unwrap_or_else(|e| serde_json::json!({"error": e.to_string()}));
    Json(result)
}
