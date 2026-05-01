use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotPhase {
    Idle,
    Prompt,
    Generating,
    Done,
}

#[derive(Debug, Clone)]
pub struct ServerSlot {
    pub id: usize,
    pub active_task_id: Option<Uuid>,
    pub phase: SlotPhase,
}

impl ServerSlot {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            active_task_id: None,
            phase: SlotPhase::Idle,
        }
    }
}
