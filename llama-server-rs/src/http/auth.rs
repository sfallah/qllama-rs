use crate::http::error::AppError;
use crate::state::AppState;
use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;

const PUBLIC_ENDPOINTS: &[&str] = &["/health", "/v1/health", "/models", "/v1/models"];

pub async fn api_key_middleware(
    axum::extract::State(state): axum::extract::State<AppState>,
    request: Request,
    next: Next,
) -> Result<Response, AppError> {
    if state.config.api_keys.is_empty() {
        return Ok(next.run(request).await);
    }

    let path = request.uri().path();
    if PUBLIC_ENDPOINTS.contains(&path) {
        return Ok(next.run(request).await);
    }

    let headers = request.headers();
    let auth_header = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.strip_prefix("Bearer ").unwrap_or(s).to_string())
        .or_else(|| {
            headers
                .get("x-api-key")
                .and_then(|v| v.to_str().ok())
                .map(ToString::to_string)
        });

    let Some(key) = auth_header else {
        return Err(AppError::Unauthorized);
    };

    if !state.config.api_keys.iter().any(|k| k == &key) {
        return Err(AppError::Unauthorized);
    }

    Ok(next.run(request).await)
}
