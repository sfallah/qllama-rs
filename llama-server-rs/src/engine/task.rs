use serde_json::{json, Value};
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub enum TaskKind {
    Completion,
    CompletionsOai,
    ChatCompletions,
    Responses,
    Transcriptions,
    AnthropicMessages,
    AnthropicCountTokens,
    Infill,
    Embeddings,
    Rerank,
    Tokenize,
    Detokenize,
    ApplyTemplate,
    LoraGet,
    LoraPost,
    SlotsGet,
    SlotsPost,
    ModelsLoad,
    ModelsUnload,
}

#[derive(Debug)]
pub struct ServerTask {
    pub id: Uuid,
    pub kind: TaskKind,
    pub payload: Value,
    pub created_at: Instant,
    pub cancel: CancellationToken,
    pub result_tx: mpsc::Sender<TaskResult>,
}

#[derive(Debug)]
pub struct TaskError {
    pub code: u16,
    pub error_type: &'static str,
    pub message: String,
}

impl TaskError {
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            code: 400,
            error_type: "invalid_request_error",
            message: message.into(),
        }
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self {
            code: 503,
            error_type: "unavailable_error",
            message: message.into(),
        }
    }

    pub fn server(message: impl Into<String>) -> Self {
        Self {
            code: 500,
            error_type: "server_error",
            message: message.into(),
        }
    }

    pub fn not_supported(message: impl Into<String>) -> Self {
        Self {
            code: 501,
            error_type: "not_supported_error",
            message: message.into(),
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "code": self.code,
            "message": self.message,
            "type": self.error_type,
        })
    }
}

#[derive(Debug)]
pub enum TaskResult {
    Chunk(Value),
    Done(Value),
    Error(TaskError),
}

#[derive(Debug)]
pub struct TaskHandle {
    pub id: Uuid,
    pub cancel: CancellationToken,
    result_rx: mpsc::Receiver<TaskResult>,
}

impl TaskHandle {
    pub fn new(id: Uuid, cancel: CancellationToken, result_rx: mpsc::Receiver<TaskResult>) -> Self {
        Self {
            id,
            cancel,
            result_rx,
        }
    }

    pub async fn recv(&mut self) -> Option<TaskResult> {
        self.result_rx.recv().await
    }

    pub fn cancel(&self) {
        self.cancel.cancel();
    }
}

/// Dropping a handle cancels its task: axum drops the handler future (or the
/// response body) as soon as the client disconnects, so HTTP code must hold the
/// handle until it has received the final result.
impl Drop for TaskHandle {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}
