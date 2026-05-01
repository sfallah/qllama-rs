use crate::engine::task::TaskKind;
use crate::http::handlers::submit_and_wait;
use crate::state::AppState;
use axum::extract::{Path, State};
use axum::Json;
use serde_json::{json, Value};

pub async fn get_slots(State(state): State<AppState>) -> Json<Value> {
    let result = submit_and_wait(&state, TaskKind::SlotsGet, json!({}))
        .await
        .unwrap_or_else(|e| serde_json::json!({"error": e.to_string()}));
    Json(result)
}

pub async fn post_slot(
    State(state): State<AppState>,
    Path(id_slot): Path<i32>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let mut payload = payload;
    payload["id_slot"] = json!(id_slot);
    let result = submit_and_wait(&state, TaskKind::SlotsPost, payload)
        .await
        .unwrap_or_else(|e| serde_json::json!({"error": e.to_string()}));
    Json(result)
}
