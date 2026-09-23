use crate::config::ServerConfig;
use crate::engine::completion::{
    legacy_chunk_json, legacy_final_json, oai_chunk_json, oai_final_json, CompletionChunk,
    CompletionParams,
};
use crate::engine::metrics::SharedMetrics;
use crate::engine::runtime::{EmitOutcome, EngineRuntime};
use crate::engine::slot::{ServerSlot, SlotPhase};
use crate::engine::task::{ServerTask, TaskError, TaskKind, TaskResult};
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
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
                        let _ = task.result_tx.blocking_send(TaskResult::Error(
                            TaskError::unavailable(format!("engine init failed: {err}")),
                        ));
                    }
                    return;
                }
            };

            while let Some(task) = rx.blocking_recv() {
                if task.cancel.is_cancelled() {
                    self.metrics.inc_cancelled();
                    let _ = task
                        .result_tx
                        .blocking_send(TaskResult::Error(TaskError::server("request cancelled")));
                    continue;
                }

                if let Some(slot) = self.slots.iter_mut().find(|s| s.active_task_id.is_none()) {
                    slot.active_task_id = Some(task.id);
                    slot.phase = SlotPhase::Generating;
                }

                match task.kind {
                    TaskKind::Completion | TaskKind::CompletionsOai => {
                        run_completion(&mut runtime, &task, &self.metrics, &self.config.model_id);
                    }
                    _ => match execute_task(&mut runtime, &task) {
                        Ok(done) => {
                            let _ = task.result_tx.blocking_send(TaskResult::Done(done));
                            self.metrics.inc_completed();
                        }
                        Err(err) => {
                            let _ = task
                                .result_tx
                                .blocking_send(TaskResult::Error(execution_error(&err)));
                        }
                    },
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

fn run_completion(
    runtime: &mut EngineRuntime,
    task: &ServerTask,
    metrics: &SharedMetrics,
    model_id: &str,
) {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let n_ctx = i32::try_from(runtime.model_n_ctx()).unwrap_or(i32::MAX);
    let params =
        match CompletionParams::from_payload(&task.payload, &runtime.defaults, n_ctx, model_id) {
            Ok(params) => params,
            Err(err) => {
                let _ = task.result_tx.blocking_send(TaskResult::Error(err));
                return;
            }
        };

    let oai = matches!(task.kind, TaskKind::CompletionsOai);
    let result = {
        let mut emit = |chunk: CompletionChunk| {
            let value = if oai {
                oai_chunk_json(&chunk, &params, created)
            } else {
                legacy_chunk_json(&chunk)
            };
            if task.result_tx.blocking_send(TaskResult::Chunk(value)).is_err() {
                EmitOutcome::ReceiverGone
            } else {
                EmitOutcome::Continue
            }
        };
        runtime.completion(&params, &task.cancel, &mut emit)
    };

    match result {
        Ok(final_result) => {
            let done = if oai {
                oai_final_json(&final_result, &params, created)
            } else {
                let gen_settings = params.sampling.generation_settings_json(
                    params.n_predict,
                    n_ctx,
                    &params.stop,
                    params.stream,
                    &params.model_name,
                );
                legacy_final_json(&final_result, &params, gen_settings, params.stream)
            };
            metrics.add_prompt_tokens(u64::try_from(final_result.n_prompt).unwrap_or(0));
            metrics.add_predicted_tokens(u64::try_from(final_result.n_decoded).unwrap_or(0));
            let _ = task.result_tx.blocking_send(TaskResult::Done(done));
            metrics.inc_completed();
        }
        Err(err) => {
            if task.cancel.is_cancelled() {
                metrics.inc_cancelled();
            }
            let _ = task.result_tx.blocking_send(TaskResult::Error(err));
        }
    }
}

fn execution_error(err: &anyhow::Error) -> TaskError {
    let message = err.to_string();
    if message.starts_with("missing field:")
        || message.starts_with("missing string field:")
        || message.starts_with("missing array field:")
    {
        TaskError::invalid_request(message)
    } else {
        TaskError::server(message)
    }
}

fn execute_task(runtime: &mut EngineRuntime, task: &ServerTask) -> anyhow::Result<serde_json::Value> {
    match task.kind {
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
