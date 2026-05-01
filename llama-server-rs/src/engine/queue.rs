use crate::engine::metrics::SharedMetrics;
use crate::engine::task::ServerTask;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{mpsc, Notify};

#[derive(Debug)]
enum QueueCommand {
    Enqueue(ServerTask),
    Defer(ServerTask),
}

#[derive(Clone)]
pub struct TaskQueue {
    tx: mpsc::Sender<QueueCommand>,
}

impl TaskQueue {
    pub fn new(metrics: SharedMetrics) -> (Self, mpsc::Receiver<ServerTask>) {
        let (cmd_tx, mut cmd_rx) = mpsc::channel::<QueueCommand>(1024);
        let (engine_tx, engine_rx) = mpsc::channel::<ServerTask>(1024);
        let notify = Arc::new(Notify::new());

        tokio::spawn({
            let notify = Arc::clone(&notify);
            async move {
                let mut main_q: VecDeque<ServerTask> = VecDeque::new();
                let mut deferred_q: VecDeque<ServerTask> = VecDeque::new();

                loop {
                    tokio::select! {
                        maybe_cmd = cmd_rx.recv() => {
                            let Some(cmd) = maybe_cmd else {
                                break;
                            };
                            match cmd {
                                QueueCommand::Enqueue(task) => {
                                    metrics.inc_enqueued();
                                    main_q.push_back(task);
                                }
                                QueueCommand::Defer(task) => {
                                    deferred_q.push_back(task);
                                }
                            }
                            notify.notify_one();
                        }
                        _ = notify.notified() => {
                            while let Some(task) = main_q.pop_front().or_else(|| deferred_q.pop_front()) {
                                if engine_tx.send(task).await.is_err() {
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        });

        (Self { tx: cmd_tx }, engine_rx)
    }

    pub async fn enqueue(&self, task: ServerTask) -> anyhow::Result<()> {
        self.tx.send(QueueCommand::Enqueue(task)).await?;
        Ok(())
    }

    pub async fn defer(&self, task: ServerTask) -> anyhow::Result<()> {
        self.tx.send(QueueCommand::Defer(task)).await?;
        Ok(())
    }
}
