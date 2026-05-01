use crate::state::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde_json::json;

pub async fn get_health(State(state): State<AppState>) -> impl IntoResponse {
    if state.engine.is_ready() {
        return (StatusCode::OK, Json(json!({"status": "ok"}))).into_response();
    }
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "error": {
                "message": "Loading model",
                "type": "unavailable_error",
                "code": 503
            }
        })),
    )
        .into_response()
}
