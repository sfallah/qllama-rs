//! Client-disconnect cancellation: the engine must drop the abandoned task and
//! stay serviceable for the next request.
#![allow(missing_docs)]

mod common;

use std::time::{Duration, Instant};

use common::TestServer;
use futures::StreamExt;
use serde_json::{json, Value};

const CANCELLED: &str = "llama_server_tasks_cancelled_total";
/// Big enough that generation is still running when the client goes away: the
/// tiny test model does ~1.3k tok/s, and 2k SSE frames also overflow the socket
/// buffers, so the engine cannot race ahead to completion.
const LONG_CTX: &str = "2048";
const LONG_PREDICT: i64 = 2048;

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

fn long_request(stream: bool) -> Value {
    json!({
        "prompt": "Once upon a time",
        "n_predict": LONG_PREDICT,
        "stream": stream,
        // keep the run from ending early on an EOG token
        "ignore_eos": true,
    })
}

/// A short completion has to go through right after the abandoned one.
async fn assert_engine_is_serviceable(server: &TestServer) {
    let response = server
        .client
        .post(server.url("/completion"))
        .timeout(Duration::from_secs(15))
        .json(&json!({"prompt": "Hi", "n_predict": 4}))
        .send()
        .await
        .unwrap_or_else(|err| {
            panic!(
                "the engine did not accept a follow-up request: {err}\n{}",
                server.log_tail()
            )
        });
    assert_eq!(response.status().as_u16(), 200);
    let body: Value = response.json().await.expect("decode the follow-up body");
    assert_eq!(body["stop"], json!(true));
}

async fn wait_for_cancellation(server: &TestServer, baseline: u64) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let current = server.metric(CANCELLED).await;
        if current > baseline {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "{CANCELLED} stayed at {baseline} after the client went away"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[tokio::test]
async fn dropping_a_stream_cancels_the_task() {
    let model = model_or_skip!("dropping_a_stream_cancels_the_task");
    let server = TestServer::with_model(&model, &["--ctx-size", LONG_CTX]).await;
    let baseline = server.metric(CANCELLED).await;

    let response = server.post("/completion", &long_request(true)).await;
    assert_eq!(response.status().as_u16(), 200);

    let mut body = response.bytes_stream();
    let mut frames = 0;
    while frames < 2 {
        let chunk = body
            .next()
            .await
            .expect("the stream ended before two frames")
            .expect("read a stream chunk");
        frames += String::from_utf8_lossy(&chunk).matches("data: ").count();
    }
    drop(body);

    assert_engine_is_serviceable(&server).await;
    wait_for_cancellation(&server, baseline).await;
}

#[tokio::test]
async fn aborting_a_non_stream_request_cancels_the_task() {
    let model = model_or_skip!("aborting_a_non_stream_request_cancels_the_task");
    let server = TestServer::with_model(&model, &["--ctx-size", LONG_CTX]).await;
    let baseline = server.metric(CANCELLED).await;

    let client = server.client.clone();
    let url = server.url("/completion");
    let inflight = tokio::spawn(async move {
        client
            .post(url)
            .json(&long_request(false))
            .send()
            .await
            .map(|response| response.status())
    });

    tokio::time::sleep(Duration::from_millis(300)).await;
    inflight.abort();
    assert!(
        inflight.await.is_err(),
        "the request finished before it could be aborted"
    );

    assert_engine_is_serviceable(&server).await;
    wait_for_cancellation(&server, baseline).await;
}
