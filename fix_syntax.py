with open('/Users/sabafallah/dev/qimia_ai_new/llama-cpp-rs/llama-server-rs/src/main.rs', 'r') as f:
    text = f.read()
import re
# re-fix error_response properly
# Actually it's easier to just find the function
text = re.sub(r'fn error_response.*?fn json_response', '''fn error_response(err: HttpError) -> Response {
    (err.status, axum::response::Json(serde_json::json!({
        "error": {
            "message": err.message,
            "type": err.err_type,
            "code": err.status.as_u16()
        }
    }))).into_response()
}
fn json_response''', text, flags=re.DOTALL)
with open('/Users/sabafallah/dev/qimia_ai_new/llama-cpp-rs/llama-server-rs/src/main.rs', 'w') as f:
    f.write(text)
