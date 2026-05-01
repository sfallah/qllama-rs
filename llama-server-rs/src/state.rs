use crate::config::ServerConfig;
use crate::engine::Engine;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<ServerConfig>,
    pub engine: Engine,
    pub started_at: Instant,
}

impl AppState {
    pub fn new(config: Arc<ServerConfig>, engine: Engine) -> Self {
        Self {
            config,
            engine,
            started_at: Instant::now(),
        }
    }
}
