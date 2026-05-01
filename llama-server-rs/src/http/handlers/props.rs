use crate::state::AppState;
use axum::extract::State;
use axum::Json;
use serde_json::{json, Value};

pub async fn get_props(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "model_path": state.config.model_path,
        "total_slots": state.config.slots,
        "is_sleeping": false,
        "default_generation_settings": {
            "n_ctx": state.config.n_ctx,
            "params": {
                "seed": 0,
                "n_predict": state.config.n_predict,
                "temperature": state.config.temperature,
                "top_k": state.config.top_k,
                "top_p": state.config.top_p,
                "min_p": state.config.min_p
            }
        }
    }))
}

pub async fn post_props(State(state): State<AppState>) -> Json<Value> {
    get_props(State(state)).await
}
