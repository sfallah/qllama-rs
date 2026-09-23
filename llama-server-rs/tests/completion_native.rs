//! End-to-end tests for the legacy `/completion` pipeline.
//!
//! Every test needs a real model; when none can be resolved the test prints a
//! SKIP line and passes.
#![allow(missing_docs)]

mod common;

use common::{sse_json_frames, TestServer, TINY_CTX};
use serde_json::{json, Value};

/// Keys the reference server puts in `timings`.
const TIMINGS_KEYS: [&str; 9] = [
    "cache_n",
    "prompt_n",
    "prompt_ms",
    "prompt_per_token_ms",
    "prompt_per_second",
    "predicted_n",
    "predicted_ms",
    "predicted_per_token_ms",
    "predicted_per_second",
];

macro_rules! model_or_skip {
    ($test:literal) => {
        match common::test_model() {
            Some(model) => model,
            None => {
                eprintln!("SKIP {}: no test model available", $test);
                return;
            }
        }
    };
}

fn as_i64(value: &Value, field: &str) -> i64 {
    value
        .get(field)
        .and_then(Value::as_i64)
        .unwrap_or_else(|| panic!("{field} is not an integer in {value}"))
}

fn as_str<'a>(value: &'a Value, field: &str) -> &'a str {
    value
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("{field} is not a string in {value}"))
}

#[tokio::test]
async fn non_stream_completion_returns_the_full_legacy_object() {
    let model = model_or_skip!("non_stream_completion_returns_the_full_legacy_object");
    let server = TestServer::with_model(&model, &["--ctx-size", TINY_CTX]).await;

    let body = server
        .post_json(
            "/completion",
            &json!({"prompt": "Once upon a time", "n_predict": 8, "seed": 42, "temperature": 0}),
        )
        .await;

    assert_eq!(body["index"], json!(0));
    assert_eq!(body["id_slot"], json!(0));
    assert_eq!(body["stop"], json!(true));
    assert!(
        !as_str(&body, "content").is_empty(),
        "content is non-empty: {body}"
    );
    assert!(
        !as_str(&body, "model").is_empty(),
        "model is non-empty: {body}"
    );
    assert_eq!(body["prompt"], json!("Once upon a time"));
    assert!(body["has_new_line"].is_boolean(), "has_new_line: {body}");
    assert!(body["truncated"].is_boolean(), "truncated: {body}");
    assert!(body["stopping_word"].is_string(), "stopping_word: {body}");
    assert!(as_i64(&body, "tokens_cached") >= 0);

    let predicted = as_i64(&body, "tokens_predicted");
    assert!((1..=8).contains(&predicted), "tokens_predicted: {body}");
    assert!(as_i64(&body, "tokens_evaluated") > 0, "prompt tokens: {body}");

    let stop_type = as_str(&body, "stop_type");
    assert!(
        ["eos", "limit", "word", "none"].contains(&stop_type),
        "unknown stop_type {stop_type}"
    );
    assert!(
        matches!(stop_type, "eos" | "limit"),
        "n_predict=8 without stop words ends on eos or limit, got {stop_type}"
    );

    // `return_tokens` defaults to false.
    assert_eq!(body["tokens"], json!([]));

    let settings = body["generation_settings"]
        .as_object()
        .unwrap_or_else(|| panic!("generation_settings is not an object: {body}"));
    for key in ["seed", "temperature", "top_k", "top_p", "n_predict", "stop"] {
        assert!(settings.contains_key(key), "generation_settings.{key}");
    }
    assert_eq!(settings["seed"], json!(42));
    assert_eq!(settings["n_predict"], json!(8));

    let timings = body["timings"]
        .as_object()
        .unwrap_or_else(|| panic!("timings is not an object: {body}"));
    for key in TIMINGS_KEYS {
        assert!(timings.contains_key(key), "timings.{key}");
    }
    assert_eq!(timings["predicted_n"], json!(predicted));

    let metrics = server.get_text("/metrics").await;
    assert!(common::parse_metric(&metrics, "llamacpp:prompt_tokens_total") > 0);
    assert!(common::parse_metric(&metrics, "llamacpp:tokens_predicted_total") > 0);
}

#[tokio::test]
async fn return_tokens_controls_the_tokens_array() {
    let model = model_or_skip!("return_tokens_controls_the_tokens_array");
    let server = TestServer::with_model(&model, &["--ctx-size", TINY_CTX]).await;
    let request = json!({"prompt": "Once upon a time", "n_predict": 8, "seed": 42, "temperature": 0});

    let mut off = request.clone();
    off["return_tokens"] = json!(false);
    assert_eq!(server.post_json("/completion", &off).await["tokens"], json!([]));

    let mut on = request;
    on["return_tokens"] = json!(true);
    let body = server.post_json("/completion", &on).await;
    let tokens = body["tokens"]
        .as_array()
        .unwrap_or_else(|| panic!("tokens is not an array: {body}"));
    assert!(!tokens.is_empty(), "tokens is non-empty: {body}");
    assert!(
        tokens.iter().all(Value::is_i64),
        "tokens are integers: {body}"
    );
    // The EOG token is counted and returned when generation stops on it.
    assert_eq!(
        i64::try_from(tokens.len()).expect("token count fits in i64"),
        as_i64(&body, "tokens_predicted")
    );
}

#[tokio::test]
async fn a_stop_word_truncates_the_completion() {
    let model = model_or_skip!("a_stop_word_truncates_the_completion");
    let server = TestServer::with_model(&model, &["--ctx-size", TINY_CTX]).await;
    let request = json!({"prompt": "Once upon a time", "n_predict": 24, "temperature": 0, "seed": 7});

    let baseline = server.post_json("/completion", &request).await;
    let full = as_str(&baseline, "content").to_string();

    let words: Vec<&str> = full.split_whitespace().filter(|w| w.len() >= 3).collect();
    if words.len() < 3 {
        eprintln!("SKIP a_stop_word_truncates_the_completion: greedy output {full:?} is too short");
        return;
    }
    let word = words[words.len() / 2].to_string();

    let mut stopped_request = request;
    stopped_request["stop"] = json!([word]);
    let stopped = server.post_json("/completion", &stopped_request).await;

    assert_eq!(stopped["stop_type"], json!("word"));
    assert_eq!(stopped["stopping_word"], json!(word));
    let content = as_str(&stopped, "content");
    assert!(
        !content.contains(&word),
        "stopped content {content:?} still contains {word:?}"
    );
    assert!(
        full.starts_with(content),
        "stopped content {content:?} is not a prefix of {full:?}"
    );
    assert!(
        content.len() < full.len(),
        "stopped content {content:?} is not shorter than {full:?}"
    );
    assert!(as_i64(&stopped, "tokens_predicted") <= 24);
}

#[tokio::test]
async fn the_legacy_stream_is_sse_without_a_done_terminator() {
    let model = model_or_skip!("the_legacy_stream_is_sse_without_a_done_terminator");
    let server = TestServer::with_model(&model, &["--ctx-size", TINY_CTX]).await;
    let request = json!({"prompt": "Once upon a time", "n_predict": 8, "temperature": 0, "seed": 42});

    let mut streamed_request = request.clone();
    streamed_request["stream"] = json!(true);
    let response = server.post("/completion", &streamed_request).await;
    assert_eq!(response.status().as_u16(), 200);
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .expect("content-type header")
        .to_string();
    assert!(
        content_type.starts_with("text/event-stream"),
        "legacy stream content-type is {content_type}"
    );

    let raw = response.text().await.expect("read the SSE body");
    assert!(
        !raw.contains("[DONE]"),
        "the legacy stream must not emit a [DONE] frame:\n{raw}"
    );

    let frames = sse_json_frames(&raw);
    assert!(frames.len() >= 2, "expected chunks plus a final frame: {raw}");
    let (last, chunks) = frames.split_last().expect("at least one frame");

    for chunk in chunks {
        assert_eq!(chunk["stop"], json!(false), "partial chunk: {chunk}");
        assert_eq!(chunk["index"], json!(0));
        assert_eq!(chunk["id_slot"], json!(0));
        assert!(chunk["content"].is_string(), "chunk content: {chunk}");
        assert!(chunk["tokens"].is_array(), "chunk tokens: {chunk}");
        assert!(as_i64(chunk, "tokens_predicted") > 0);
        assert!(as_i64(chunk, "tokens_evaluated") > 0);
    }

    assert_eq!(last["stop"], json!(true));
    assert_eq!(last["content"], json!(""));
    assert_eq!(last["tokens"], json!([]));
    assert!(last["stop_type"].is_string(), "final stop_type: {last}");
    assert!(
        last["generation_settings"].is_object(),
        "final generation_settings: {last}"
    );
    let timings = last["timings"]
        .as_object()
        .unwrap_or_else(|| panic!("final timings is not an object: {last}"));
    for key in TIMINGS_KEYS {
        assert!(timings.contains_key(key), "final timings.{key}");
    }

    let streamed: String = chunks
        .iter()
        .map(|chunk| chunk["content"].as_str().unwrap_or_default())
        .collect();
    let non_stream = server.post_json("/completion", &request).await;
    assert_eq!(streamed, as_str(&non_stream, "content"));
}

/// Regression: fast tasks used to race the broadcast resubscribe and 500.
#[tokio::test]
async fn sequential_tokenize_requests_never_race() {
    let model = model_or_skip!("sequential_tokenize_requests_never_race");
    let server = TestServer::with_model(&model, &["--ctx-size", TINY_CTX]).await;

    for round in 0..20 {
        let response = server
            .post("/tokenize", &json!({"content": "hello world"}))
            .await;
        assert_eq!(response.status().as_u16(), 200, "round {round}");
        let body: Value = response.json().await.expect("decode /tokenize body");
        let tokens = body["tokens"]
            .as_array()
            .unwrap_or_else(|| panic!("round {round} has no tokens array: {body}"));
        assert!(!tokens.is_empty(), "round {round} tokenized nothing: {body}");
    }
}
