use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Default)]
pub struct EngineMetrics {
    pub enqueued_tasks: AtomicU64,
    pub completed_tasks: AtomicU64,
    pub cancelled_tasks: AtomicU64,
    pub prompt_tokens_total: AtomicU64,
    pub tokens_predicted_total: AtomicU64,
}

impl EngineMetrics {
    pub fn inc_enqueued(&self) {
        self.enqueued_tasks.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_completed(&self) {
        self.completed_tasks.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_cancelled(&self) {
        self.cancelled_tasks.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_prompt_tokens(&self, n: u64) {
        self.prompt_tokens_total.fetch_add(n, Ordering::Relaxed);
    }

    pub fn add_predicted_tokens(&self, n: u64) {
        self.tokens_predicted_total.fetch_add(n, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> (u64, u64, u64) {
        (
            self.enqueued_tasks.load(Ordering::Relaxed),
            self.completed_tasks.load(Ordering::Relaxed),
            self.cancelled_tasks.load(Ordering::Relaxed),
        )
    }

    pub fn token_snapshot(&self) -> (u64, u64) {
        (
            self.prompt_tokens_total.load(Ordering::Relaxed),
            self.tokens_predicted_total.load(Ordering::Relaxed),
        )
    }
}

pub type SharedMetrics = Arc<EngineMetrics>;
