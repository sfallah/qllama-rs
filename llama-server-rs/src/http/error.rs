use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("authentication failed")]
    Unauthorized,
    #[error("internal error: {0}")]
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, err_type, msg) = match self {
            Self::InvalidRequest(m) => (StatusCode::BAD_REQUEST, "invalid_request_error", m),
            Self::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                "authentication_error",
                "Invalid API Key".to_string(),
            ),
            Self::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, "server_error", m),
        };

        (
            status,
            Json(json!({
                "error": {
                    "message": msg,
                    "type": err_type,
                    "code": status.as_u16()
                }
            })),
        )
            .into_response()
    }
}

pub type AppResult<T> = Result<T, AppError>;
