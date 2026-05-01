use serde_json::Value;

#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub seed: u32,
}

impl SamplingConfig {
    pub fn from_payload(payload: &Value) -> Self {
        Self {
            temperature: payload
                .get("temperature")
                .and_then(Value::as_f64)
                .unwrap_or(1.0) as f32,
            top_p: payload.get("top_p").and_then(Value::as_f64).unwrap_or(1.0) as f32,
            top_k: payload.get("top_k").and_then(Value::as_i64).unwrap_or(0) as i32,
            seed: payload.get("seed").and_then(Value::as_u64).unwrap_or(0) as u32,
        }
    }
}
