pub mod auth;
pub mod cors;
pub mod error;
pub mod handlers;
pub mod sse;

use crate::http::handlers::apply_template::post_apply_template;
use crate::http::handlers::chat::post_chat_completions;
use crate::http::handlers::completions::{
    post_anthropic_count_tokens, post_anthropic_messages, post_completion, post_completions_oai,
    post_responses,
};
use crate::http::handlers::embeddings::post_embeddings;
use crate::http::handlers::health::get_health;
use crate::http::handlers::infill::post_infill;
use crate::http::handlers::lora::{get_lora, post_lora};
use crate::http::handlers::metrics::get_metrics;
use crate::http::handlers::models::get_models;
use crate::http::handlers::props::{get_props, post_props};
use crate::http::handlers::rerank::post_rerank;
use crate::http::handlers::router::{post_models_load, post_models_unload};
use crate::http::handlers::slots::{get_slots, post_slot};
use crate::http::handlers::tokenize::{post_detokenize, post_tokenize};
use crate::http::handlers::transcriptions::post_transcriptions;
use crate::state::AppState;
use axum::http::StatusCode;
use axum::middleware;
use axum::routing::{get, post};
use axum::Router;
use tower::ServiceBuilder;
use tower_http::request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer};
use tower_http::trace::TraceLayer;

pub fn build_router(state: AppState) -> Router {
    let x_request_id = axum::http::HeaderName::from_static("x-request-id");

    Router::new()
        .route("/health", get(get_health))
        .route("/v1/health", get(get_health))
        .route("/models", get(get_models))
        .route("/v1/models", get(get_models))
        .route("/metrics", get(get_metrics))
        .route("/props", get(get_props).post(post_props))
        .route("/completion", post(post_completion))
        .route("/completions", post(post_completion))
        .route("/v1/completions", post(post_completions_oai))
        .route("/chat/completions", post(post_chat_completions))
        .route("/v1/chat/completions", post(post_chat_completions))
        .route("/responses", post(post_responses))
        .route("/v1/responses", post(post_responses))
        .route("/audio/transcriptions", post(post_transcriptions))
        .route("/v1/audio/transcriptions", post(post_transcriptions))
        .route("/v1/messages", post(post_anthropic_messages))
        .route(
            "/v1/messages/count_tokens",
            post(post_anthropic_count_tokens),
        )
        .route("/infill", post(post_infill))
        .route("/embedding", post(post_embeddings))
        .route("/embeddings", post(post_embeddings))
        .route("/v1/embeddings", post(post_embeddings))
        .route("/rerank", post(post_rerank))
        .route("/reranking", post(post_rerank))
        .route("/v1/rerank", post(post_rerank))
        .route("/v1/reranking", post(post_rerank))
        .route("/tokenize", post(post_tokenize))
        .route("/detokenize", post(post_detokenize))
        .route("/apply-template", post(post_apply_template))
        .route("/lora-adapters", get(get_lora).post(post_lora))
        .route("/slots", get(get_slots))
        .route("/slots/:id_slot", post(post_slot))
        .route("/models/load", post(post_models_load))
        .route("/models/unload", post(post_models_unload))
        .route("/*path", axum::routing::options(options_handler))
        .layer(
            ServiceBuilder::new()
                .layer(SetRequestIdLayer::new(
                    x_request_id.clone(),
                    MakeRequestUuid,
                ))
                .layer(PropagateRequestIdLayer::new(x_request_id))
                .layer(TraceLayer::new_for_http())
                .layer(crate::http::cors::cors_layer()),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            crate::http::auth::api_key_middleware,
        ))
        .with_state(state)
}

async fn options_handler() -> StatusCode {
    StatusCode::NO_CONTENT
}
