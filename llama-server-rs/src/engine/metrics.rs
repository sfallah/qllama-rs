use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Default)]
pub struct EngineMetrics {
    pub enqueued_tasks: AtomicU64,
    pub completed_tasks: AtomicU64,
    pub cancelled_tasks: AtomicU64,
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

    pub fn snapshot(&self) -> (u64, u64, u64) {
        (
            self.enqueued_tasks.load(Ordering::Relaxed),
            self.completed_tasks.load(Ordering::Relaxed),
            self.cancelled_tasks.load(Ordering::Relaxed),
        )
    }
}

pub type SharedMetrics = Arc<EngineMetrics>;
