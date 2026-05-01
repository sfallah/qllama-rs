use crate::state::AppState;
use axum::extract::State;
use axum::Json;
use serde_json::json;

pub async fn get_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.config.model_id,
            "object": "model",
            "owned_by": "llama-server-rs",
            "created": 0,
            "permission": [],
        }]
    }))
}
