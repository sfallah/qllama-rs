use crate::state::AppState;
use axum::extract::State;

pub async fn get_metrics(State(state): State<AppState>) -> String {
    let uptime = state.started_at.elapsed().as_secs();
    let (enqueued, completed, cancelled) = state.engine.metrics_snapshot();

    format!(
        "llama_server_uptime_seconds {uptime}\nllama_server_tasks_enqueued_total {enqueued}\nllama_server_tasks_completed_total {completed}\nllama_server_tasks_cancelled_total {cancelled}\n"
    )
}
