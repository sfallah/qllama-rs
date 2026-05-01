use serde_json::Value;
use std::time::Instant;
use tokio::sync::broadcast;
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
    pub result_tx: broadcast::Sender<TaskResult>,
}

#[derive(Debug, Clone)]
pub enum TaskResult {
    Chunk(Value),
    Done(Value),
    Error(String),
}

pub struct TaskHandle {
    pub id: Uuid,
    pub cancel: CancellationToken,
    pub result_rx: broadcast::Receiver<TaskResult>,
}

impl TaskHandle {
    pub fn cancel(&self) {
        self.cancel.cancel();
    }
}
