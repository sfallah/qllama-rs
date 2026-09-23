use crate::engine::task::TaskError;
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
    #[error("not supported: {0}")]
    NotSupported(String),
    #[error("unavailable: {0}")]
    Unavailable(String),
    #[error("timeout: {0}")]
    Timeout(String),
    #[error("internal error: {0}")]
    Internal(String),
}

impl From<TaskError> for AppError {
    fn from(err: TaskError) -> Self {
        match err.code {
            400 => Self::InvalidRequest(err.message),
            501 => Self::NotSupported(err.message),
            503 => Self::Unavailable(err.message),
            504 => Self::Timeout(err.message),
            _ => Self::Internal(err.message),
        }
    }
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
            Self::NotSupported(m) => (StatusCode::NOT_IMPLEMENTED, "not_supported_error", m),
            Self::Unavailable(m) => (StatusCode::SERVICE_UNAVAILABLE, "unavailable_error", m),
            Self::Timeout(m) => (StatusCode::GATEWAY_TIMEOUT, "timeout_error", m),
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
