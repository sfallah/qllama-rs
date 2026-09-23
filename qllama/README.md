# qllama

Safe Rust bindings to [llama.cpp](https://github.com/ggml-org/llama.cpp), the API crate of
[qllama-rs](https://github.com/sfallah/qllama-rs). The raw bindings and the llama.cpp build live in
`qllama-sys`.

The crate wraps llama.cpp's C API closely — models, contexts, batches, sampling, embeddings and
pooling (including reranking), grammars, the OpenAI-compatible chat-template and tool-calling flow,
and multimodal input behind the `mtmd` feature. It is derived from
[utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs) (`llama-cpp-2`) and is not
API-compatible with it.

## Example: OpenAI-style tool calling

```rust
use qllama::openai::OpenAIChatTemplateParams;
use serde_json::json;

let template = model.chat_template(None)?;

let tools_json = json!([{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Fetch current weather by city.",
        "parameters": {
            "type": "object",
            "properties": { "location": { "type": "string" } },
            "required": ["location"]
        }
    }
}])
.to_string();

let messages_json = json!([
    { "role": "system", "content": "You are a tool caller." },
    { "role": "user", "content": "Fetch the weather in Paris." }
])
.to_string();

let params = OpenAIChatTemplateParams {
    messages_json: &messages_json,
    tools_json: Some(&tools_json),
    tool_choice: Some("auto"),
    json_schema: None,
    grammar: None,
    reasoning_format: None,
    chat_template_kwargs: Some("{}"),
    add_generation_prompt: true,
    use_jinja: true,
    parallel_tool_calls: false,
    enable_thinking: false,
    add_bos: false,
    add_eos: false,
    parse_tool_calls: true,
};

let result = model.apply_chat_template_oaicompat(&template, &params)?;
```

For grammar generation from a JSON schema string on its own, use `qllama::json_schema_to_grammar`.

## Dependencies

`qllama-sys` runs bindgen at build time, so clang must be installed; see the
[bindgen requirements](https://rust-lang.github.io/rust-bindgen/requirements.html). It also builds
llama.cpp with CMake, which needs `cmake` and a C/C++ toolchain.

## Safety

The wrapper prevents the common misuses of the llama.cpp API, but it is a thin layer over a large C
library and there are ways to reach undefined behaviour through it. Please open an issue if you find
one. Prefer a higher-level abstraction on top of this crate for application code.

## Contributing

Contributions are welcome. Open an issue before starting a non-trivial change.
