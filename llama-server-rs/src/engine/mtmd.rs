use serde_json::Value;

pub fn extract_multimodal_payload(payload: &Value) -> Option<Value> {
    payload.get("multimodal_data").cloned()
}
