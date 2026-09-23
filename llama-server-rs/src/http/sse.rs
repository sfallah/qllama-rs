use crate::engine::task::{TaskHandle, TaskResult};
use axum::body::Body;
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
use axum::response::Response;
use bytes::Bytes;
use serde_json::{json, Value};
use std::convert::Infallible;

const DONE_FRAME: &[u8] = b"data: [DONE]\n\n";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamFraming {
    LegacySse,
    OaiSse,
}

/// Streams a completion task as server-sent events. The generator owns the task
/// handle, so a client disconnect drops the body and cancels the task.
pub fn stream_completion_response(
    first: TaskResult,
    handle: TaskHandle,
    framing: StreamFraming,
) -> Response {
    let body = async_stream::stream! {
        let mut handle = handle;
        let mut pending = Some(first);
        loop {
            let result = match pending.take() {
                Some(result) => Some(result),
                None => handle.recv().await,
            };
            match result {
                Some(TaskResult::Chunk(value)) => yield Ok::<Bytes, Infallible>(frame(&value)),
                Some(TaskResult::Done(value)) => {
                    yield Ok::<Bytes, Infallible>(frame(&value));
                    if framing == StreamFraming::OaiSse {
                        yield Ok::<Bytes, Infallible>(Bytes::from_static(DONE_FRAME));
                    }
                    break;
                }
                Some(TaskResult::Error(err)) => {
                    yield Ok::<Bytes, Infallible>(frame(&json!({"error": err.to_json()})));
                    break;
                }
                None => break,
            }
        }
    };

    Response::builder()
        .header(CONTENT_TYPE, "text/event-stream")
        .header(CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(body))
        .expect("event-stream response headers are valid")
}

fn frame(value: &Value) -> Bytes {
    Bytes::from(format!("data: {value}\n\n"))
}
