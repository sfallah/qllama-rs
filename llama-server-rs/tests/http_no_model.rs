//! HTTP-surface tests that need no model: readiness gating, the error envelope
//! and API-key handling. These always run.
#![allow(missing_docs)]

mod common;

use common::TestServer;
use serde_json::{json, Value};

/// The exact envelope the reference server returns while the model is not up.
fn loading_envelope() -> Value {
    json!({
        "error": {
            "code": 503,
            "message": "Loading model",
            "type": "unavailable_error"
        }
    })
}

#[tokio::test]
async fn health_reports_loading_model_until_the_engine_is_ready() {
    let server = TestServer::without_model(&[]).await;

    let response = server
        .client
        .get(server.url("/health"))
        .send()
        .await
        .expect("GET /health");

    assert_eq!(response.status().as_u16(), 503);
    let body: Value = response.json().await.expect("decode /health body");
    assert_eq!(body, loading_envelope());
}

#[tokio::test]
async fn completion_reports_loading_model_until_the_engine_is_ready() {
    let server = TestServer::without_model(&[]).await;

    let response = server.post("/completion", &json!({"prompt": "hi"})).await;

    assert_eq!(response.status().as_u16(), 503);
    let body: Value = response.json().await.expect("decode /completion body");
    assert_eq!(body, loading_envelope());
}

#[tokio::test]
async fn completion_without_an_api_key_is_rejected() {
    let server = TestServer::without_model(&["--api-key", "secret"]).await;

    let response = server.post("/completion", &json!({"prompt": "hi"})).await;

    assert_eq!(response.status().as_u16(), 401);
    let body: Value = response.json().await.expect("decode /completion body");
    assert_eq!(body["error"]["code"], json!(401));
    assert_eq!(body["error"]["type"], json!("authentication_error"));
    assert!(
        body["error"]["message"].as_str().is_some_and(|m| !m.is_empty()),
        "error envelope carries a message: {body}"
    );
}

#[tokio::test]
async fn health_stays_public_when_an_api_key_is_configured() {
    let server = TestServer::without_model(&["--api-key", "secret"]).await;

    let response = server
        .client
        .get(server.url("/health"))
        .send()
        .await
        .expect("GET /health");

    let status = response.status().as_u16();
    assert_ne!(status, 401, "/health must not be behind the API key");
    assert_eq!(status, 503, "the model never loads, so /health stays 503");
}

#[tokio::test]
async fn malformed_json_is_a_bad_request() {
    let server = TestServer::without_model(&["--api-key", "secret"]).await;

    let response = server
        .client
        .post(server.url("/completion"))
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .bearer_auth("secret")
        .body("{not json")
        .send()
        .await
        .expect("POST /completion");

    assert_eq!(response.status().as_u16(), 400);
}
