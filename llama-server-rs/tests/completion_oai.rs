//! End-to-end tests for the OpenAI-compatible `/v1/completions` pipeline.
#![allow(missing_docs)]

mod common;

use common::{TestServer, TINY_CTX};
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde_json::{json, Value};

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

fn as_i64(value: &Value, pointer: &str) -> i64 {
    value
        .pointer(pointer)
        .and_then(Value::as_i64)
        .unwrap_or_else(|| panic!("{pointer} is not an integer in {value}"))
}

#[tokio::test]
async fn non_stream_oai_completion_matches_the_openai_shape() {
    let model = model_or_skip!("non_stream_oai_completion_matches_the_openai_shape");
    let server = TestServer::with_model(&model, &["--ctx-size", TINY_CTX]).await;

    let body = server
        .post_json(
            "/v1/completions",
            &json!({"prompt": "Once upon a time", "max_tokens": 8, "seed": 42, "temperature": 0}),
        )
        .await;

    assert_eq!(body["object"], json!("text_completion"));
    let id = body["id"]
        .as_str()
        .unwrap_or_else(|| panic!("id is not a string: {body}"));
    assert!(id.starts_with("chatcmpl-"), "unexpected id {id}");
    assert!(as_i64(&body, "/created") > 0, "created: {body}");
    assert!(
        body["system_fingerprint"]
            .as_str()
            .is_some_and(|f| !f.is_empty()),
        "system_fingerprint: {body}"
    );
    assert!(
        body["model"].as_str().is_some_and(|m| !m.is_empty()),
        "model: {body}"
    );

    let choices = body["choices"]
        .as_array()
        .unwrap_or_else(|| panic!("choices is not an array: {body}"));
    assert_eq!(choices.len(), 1);
    let choice = &choices[0];
    assert!(
        choice["text"].as_str().is_some_and(|t| !t.is_empty()),
        "choice text: {choice}"
    );
    assert_eq!(choice["index"], json!(0));
    assert_eq!(choice["logprobs"], Value::Null);
    let finish_reason = choice["finish_reason"]
        .as_str()
        .unwrap_or_else(|| panic!("finish_reason is not a string: {choice}"));
    assert!(
        ["stop", "length"].contains(&finish_reason),
        "unexpected finish_reason {finish_reason}"
    );

    let completion_tokens = as_i64(&body, "/usage/completion_tokens");
    let prompt_tokens = as_i64(&body, "/usage/prompt_tokens");
    assert_eq!(
        completion_tokens + prompt_tokens,
        as_i64(&body, "/usage/total_tokens")
    );
    assert!(
        completion_tokens <= 8,
        "max_tokens=8 must cap completion_tokens, got {completion_tokens}"
    );
    assert!(
        body.pointer("/usage/prompt_tokens_details").is_some(),
        "usage.prompt_tokens_details: {body}"
    );

    // A tighter budget is honoured too.
    let capped = server
        .post_json(
            "/v1/completions",
            &json!({"prompt": "Once upon a time", "max_tokens": 3, "seed": 42, "temperature": 0}),
        )
        .await;
    assert!(as_i64(&capped, "/usage/completion_tokens") <= 3);
}

#[tokio::test]
async fn the_oai_stream_ends_with_usage_then_done() {
    let model = model_or_skip!("the_oai_stream_ends_with_usage_then_done");
    let server = TestServer::with_model(&model, &["--ctx-size", TINY_CTX]).await;
    let request = json!({"prompt": "Once upon a time", "max_tokens": 8, "seed": 42, "temperature": 0});

    let mut streamed_request = request.clone();
    streamed_request["stream"] = json!(true);
    let response = server.post("/v1/completions", &streamed_request).await;
    assert_eq!(response.status().as_u16(), 200);
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .expect("content-type header")
        .to_string();
    assert!(
        content_type.starts_with("text/event-stream"),
        "OAI stream content-type is {content_type}"
    );

    let mut events = response.bytes_stream().eventsource();
    let mut payloads = Vec::new();
    while let Some(event) = events.next().await {
        payloads.push(event.expect("SSE event").data);
    }

    assert_eq!(
        payloads.last().map(String::as_str),
        Some("[DONE]"),
        "the OAI stream must end with [DONE]: {payloads:?}"
    );
    let frames: Vec<Value> = payloads[..payloads.len() - 1]
        .iter()
        .map(|data| {
            serde_json::from_str(data)
                .unwrap_or_else(|err| panic!("SSE frame is not JSON ({err}): {data}"))
        })
        .collect();
    assert!(frames.len() >= 2, "expected deltas plus a final frame");

    let (last, deltas) = frames.split_last().expect("at least one frame");
    for delta in deltas {
        assert_eq!(delta["object"], json!("text_completion"));
        assert_eq!(delta["choices"][0]["finish_reason"], Value::Null);
        assert_eq!(delta["choices"][0]["index"], json!(0));
        assert!(
            delta["choices"][0]["text"].is_string(),
            "delta text: {delta}"
        );
        assert!(
            delta.get("usage").is_none(),
            "partial chunks carry no usage: {delta}"
        );
    }

    assert_eq!(last["choices"][0]["text"], json!(""));
    assert!(
        last["choices"][0]["finish_reason"].is_string(),
        "final finish_reason: {last}"
    );
    assert_eq!(
        as_i64(last, "/usage/completion_tokens") + as_i64(last, "/usage/prompt_tokens"),
        as_i64(last, "/usage/total_tokens")
    );

    let streamed: String = deltas
        .iter()
        .map(|delta| delta["choices"][0]["text"].as_str().unwrap_or_default())
        .collect();
    let non_stream = server.post_json("/v1/completions", &request).await;
    assert_eq!(
        streamed,
        non_stream["choices"][0]["text"]
            .as_str()
            .unwrap_or_else(|| panic!("non-stream text: {non_stream}"))
    );
}
