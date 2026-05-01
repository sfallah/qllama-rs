use crate::engine::slot::ServerSlot;

pub fn active_slots(slots: &[ServerSlot]) -> impl Iterator<Item=&ServerSlot> {
    slots.iter().filter(|s| s.active_task_id.is_some())
}
