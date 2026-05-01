use crate::config::ServerConfig;
use crate::engine::metrics::SharedMetrics;
use crate::engine::runtime::EngineRuntime;
use crate::engine::slot::{ServerSlot, SlotPhase};
use crate::engine::task::{ServerTask, TaskKind, TaskResult};
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::mpsc;

pub struct EngineLoop {
    slots: Vec<ServerSlot>,
    metrics: SharedMetrics,
    config: Arc<ServerConfig>,
    ready: Arc<AtomicBool>,
}

impl EngineLoop {
    pub fn new(config: Arc<ServerConfig>, metrics: SharedMetrics, ready: Arc<AtomicBool>) -> Self {
        let mut slots = Vec::with_capacity(config.slots);
        for i in 0..config.slots {
            slots.push(ServerSlot::new(i));
        }
        Self {
            slots,
            metrics,
            config,
            ready,
        }
    }

    pub fn spawn(mut self, mut rx: mpsc::Receiver<ServerTask>) {
        tokio::task::spawn_blocking(move || {
            let mut runtime = match EngineRuntime::new(&self.config) {
                Ok(runtime) => {
                    self.ready.store(true, Ordering::Relaxed);
                    runtime
                }
                Err(err) => {
                    while let Some(task) = rx.blocking_recv() {
                        let _ = task
                            .result_tx
                            .send(TaskResult::Error(format!("engine init failed: {err}")));
                    }
                    return;
                }
            };

            while let Some(task) = rx.blocking_recv() {
                if task.cancel.is_cancelled() {
                    self.metrics.inc_cancelled();
                    let _ = task
                        .result_tx
                        .send(TaskResult::Error("request cancelled".to_string()));
                    continue;
                }

                if let Some(slot) = self.slots.iter_mut().find(|s| s.active_task_id.is_none()) {
                    slot.active_task_id = Some(task.id);
                    slot.phase = SlotPhase::Generating;
                }

                let result = execute_task(&mut runtime, &task);
                match result {
                    Ok(done) => {
                        let _ = task.result_tx.send(TaskResult::Done(done));
                        self.metrics.inc_completed();
                    }
                    Err(err) => {
                        let _ = task.result_tx.send(TaskResult::Error(err.to_string()));
                    }
                }

                for slot in &mut self.slots {
                    if slot.active_task_id == Some(task.id) {
                        slot.active_task_id = None;
                        slot.phase = SlotPhase::Done;
                        slot.phase = SlotPhase::Idle;
                    }
                }
            }
        });
    }
}

fn execute_task(runtime: &mut EngineRuntime, task: &ServerTask) -> anyhow::Result<serde_json::Value> {
    match task.kind {
        TaskKind::Completion | TaskKind::CompletionsOai => runtime.completion(&task.payload),
        TaskKind::Tokenize => runtime.tokenize(&task.payload),
        TaskKind::Detokenize => runtime.detokenize(&task.payload),
        TaskKind::ApplyTemplate => runtime.apply_template(&task.payload),
        _ => Ok(json!({
            "id": task.id.to_string(),
            "status": "not_implemented",
            "kind": format!("{:?}", task.kind),
        })),
    }
}
