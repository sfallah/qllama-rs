use serde_json::Value;

pub fn normalize_messages(payload: &Value) -> Value {
    payload.get("messages").cloned().unwrap_or(Value::Null)
}
