pub mod batch;
pub mod chat;
pub mod r#loop;
pub mod metrics;
pub mod mtmd;
pub mod prompt_cache;
pub mod queue;
pub mod runtime;
pub mod sampler;
pub mod slot;
pub mod task;

use crate::engine::metrics::{EngineMetrics, SharedMetrics};
use crate::engine::queue::TaskQueue;
use crate::engine::r#loop::EngineLoop;
use crate::engine::task::{ServerTask, TaskHandle, TaskKind, TaskResult};
use crate::config::ServerConfig;
use serde_json::Value;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[derive(Clone)]
pub struct Engine {
    queue: TaskQueue,
    metrics: SharedMetrics,
    ready: Arc<AtomicBool>,
}

impl Engine {
    pub fn new(config: Arc<ServerConfig>) -> Self {
        let metrics = Arc::new(EngineMetrics::default());
        let (queue, rx) = TaskQueue::new(metrics.clone());
        let ready = Arc::new(AtomicBool::new(false));
        EngineLoop::new(config, metrics.clone(), ready.clone()).spawn(rx);
        Self { queue, metrics, ready }
    }

    pub async fn submit(&self, kind: TaskKind, payload: Value) -> anyhow::Result<TaskHandle> {
        let id = Uuid::new_v4();
        let cancel = CancellationToken::new();
        let (result_tx, result_rx) = broadcast::channel::<TaskResult>(128);

        let task = ServerTask {
            id,
            kind,
            payload,
            created_at: Instant::now(),
            cancel: cancel.clone(),
            result_tx,
        };

        self.queue.enqueue(task).await?;

        Ok(TaskHandle {
            id,
            cancel,
            result_rx,
        })
    }

    pub fn metrics_snapshot(&self) -> (u64, u64, u64) {
        self.metrics.snapshot()
    }

    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
    }
}
