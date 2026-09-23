use crate::engine::sampler::{json_bool, json_i32, json_i64, SamplingDefaults, SamplingParams};
use crate::engine::task::TaskError;
use serde_json::{json, Map, Value};
use uuid::Uuid;

pub const SYSTEM_FINGERPRINT: &str = concat!("llama-server-rs-", env!("CARGO_PKG_VERSION"));

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptInput {
    Text(String),
    Tokens(Vec<i32>),
}

#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct CompletionParams {
    pub prompt: PromptInput,
    pub n_predict: i32,
    pub t_max_predict_ms: i64,
    pub return_tokens: bool,
    pub stream: bool,
    pub timings_per_token: bool,
    pub cache_prompt: bool,
    pub stop: Vec<String>,
    pub sampling: SamplingParams,
    pub model_name: String,
    pub cmpl_id: String,
}

impl CompletionParams {
    pub fn from_payload(
        payload: &Value,
        defaults: &SamplingDefaults,
        n_ctx: i32,
        model_id: &str,
    ) -> Result<Self, TaskError> {
        let prompt = parse_prompt(payload)?;
        let stop = parse_stop(payload)?;
        let sampling = SamplingParams::from_payload(payload, defaults, n_ctx)?;

        let max_tokens = json_i32(payload, "max_tokens", defaults.n_predict);
        let max_completion_tokens = json_i32(payload, "max_completion_tokens", max_tokens);
        let n_predict = json_i32(payload, "n_predict", max_completion_tokens);

        let model_name = payload
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(model_id)
            .to_string();

        Ok(Self {
            prompt,
            n_predict,
            t_max_predict_ms: json_i64(payload, "t_max_predict_ms", -1),
            return_tokens: json_bool(payload, "return_tokens", false),
            stream: json_bool(payload, "stream", false),
            timings_per_token: json_bool(payload, "timings_per_token", false),
            cache_prompt: json_bool(payload, "cache_prompt", true),
            stop,
            sampling,
            model_name,
            cmpl_id: format!("chatcmpl-{}", Uuid::new_v4().simple()),
        })
    }
}

fn parse_prompt(payload: &Value) -> Result<PromptInput, TaskError> {
    match payload.get("prompt") {
        Some(Value::String(text)) => Ok(PromptInput::Text(text.clone())),
        Some(Value::Array(items)) => {
            if items.is_empty() {
                return Err(TaskError::invalid_request("\"prompt\" must not be empty"));
            }
            let mut tokens = Vec::with_capacity(items.len());
            for item in items {
                match item.as_i64() {
                    Some(token) => tokens.push(clamp_token(token)),
                    None => {
                        return Err(TaskError::not_supported(
                            "multiple prompts not supported yet",
                        ))
                    }
                }
            }
            Ok(PromptInput::Tokens(tokens))
        }
        None | Some(Value::Null) => Err(TaskError::invalid_request("\"prompt\" is required")),
        Some(_) => Err(TaskError::invalid_request(
            "\"prompt\" must be a string or a list of tokens",
        )),
    }
}

fn clamp_token(token: i64) -> i32 {
    i32::try_from(token).unwrap_or(if token < 0 { i32::MIN } else { i32::MAX })
}

fn parse_stop(payload: &Value) -> Result<Vec<String>, TaskError> {
    let Some(Value::Array(items)) = payload.get("stop") else {
        return Ok(Vec::new());
    };
    let mut stop = Vec::with_capacity(items.len());
    for item in items {
        match item {
            Value::String(word) => {
                if !word.is_empty() {
                    stop.push(word.clone());
                }
            }
            Value::Null => {}
            Value::Array(inner) if inner.is_empty() => {}
            Value::Object(inner) if inner.is_empty() => {}
            _ => {
                return Err(TaskError::invalid_request(
                    "\"stop\" must be an array of strings",
                ))
            }
        }
    }
    Ok(stop)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StopType {
    #[default]
    None,
    Eos,
    Limit,
    Word,
}

impl StopType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Eos => "eos",
            Self::Limit => "limit",
            Self::Word => "word",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopScan {
    None,
    Full { pos: usize, word: usize },
    Partial { pos: usize },
}

#[derive(Debug, Clone, Default)]
pub struct StopMatcher {
    stops: Vec<String>,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        Self { stops }
    }

    pub fn stops(&self) -> &[String] {
        &self.stops
    }

    pub fn scan(&self, text: &str, last_token_len: usize, allow_partial: bool) -> StopScan {
        let haystack = text.as_bytes();

        let mut best: Option<(usize, usize)> = None;
        for (index, word) in self.stops.iter().enumerate() {
            let needle = word.as_bytes();
            if needle.is_empty() {
                continue;
            }
            let window = needle.len() + last_token_len;
            let from = haystack.len().saturating_sub(window);
            if let Some(pos) = find_from(haystack, needle, from) {
                let better = match best {
                    Some((best_pos, _)) => pos < best_pos,
                    None => true,
                };
                if better {
                    best = Some((pos, index));
                }
            }
        }
        if let Some((pos, word)) = best {
            return StopScan::Full { pos, word };
        }

        if allow_partial {
            let mut best_pos: Option<usize> = None;
            for word in &self.stops {
                if let Some(pos) = find_partial_stop(haystack, word.as_bytes()) {
                    let better = match best_pos {
                        Some(current) => pos < current,
                        None => true,
                    };
                    if better {
                        best_pos = Some(pos);
                    }
                }
            }
            if let Some(pos) = best_pos {
                return StopScan::Partial { pos };
            }
        }

        StopScan::None
    }
}

fn find_from(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if from >= haystack.len() {
        return None;
    }
    haystack[from..]
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|pos| pos + from)
}

fn find_partial_stop(text: &[u8], stop: &[u8]) -> Option<usize> {
    if text.is_empty() || stop.is_empty() {
        return None;
    }
    let max_len = text.len().min(stop.len());
    let last = text[text.len() - 1];
    for len in (1..=max_len).rev() {
        if stop[len - 1] == last && text.ends_with(&stop[..len]) {
            return Some(text.len() - len);
        }
    }
    None
}

#[derive(Debug, Default)]
pub struct IncrementalUtf8 {
    pending: Vec<u8>,
}

impl IncrementalUtf8 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, bytes: &[u8]) -> Option<String> {
        self.pending.extend_from_slice(bytes);
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(_) => break,
                Err(error) => match error.error_len() {
                    None => return None,
                    Some(len) => {
                        let start = error.valid_up_to();
                        self.pending
                            .splice(start..start + len, "\u{fffd}".bytes())
                            .for_each(drop);
                    }
                },
            }
        }
        if self.pending.is_empty() {
            return None;
        }
        String::from_utf8(std::mem::take(&mut self.pending)).ok()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

#[derive(Debug, Clone, Default)]
pub struct Timings {
    pub prompt_n: i32,
    pub prompt_ms: f64,
    pub predicted_n: i32,
    pub predicted_ms: f64,
}

impl Timings {
    pub fn to_json(&self) -> Value {
        json!({
            "cache_n": 0,
            "prompt_n": self.prompt_n,
            "prompt_ms": self.prompt_ms,
            "prompt_per_token_ms": per_token_ms(self.prompt_ms, self.prompt_n),
            "prompt_per_second": per_second(self.prompt_ms, self.prompt_n),
            "predicted_n": self.predicted_n,
            "predicted_ms": self.predicted_ms,
            "predicted_per_token_ms": per_token_ms(self.predicted_ms, self.predicted_n),
            "predicted_per_second": per_second(self.predicted_ms, self.predicted_n),
        })
    }
}

fn per_token_ms(ms: f64, n: i32) -> f64 {
    if n > 0 {
        ms / f64::from(n)
    } else {
        0.0
    }
}

fn per_second(ms: f64, n: i32) -> f64 {
    if ms > 0.0 {
        1e3 / ms * f64::from(n)
    } else {
        0.0
    }
}

#[derive(Debug, Clone)]
pub struct CompletionChunk {
    pub content: String,
    pub token: i32,
    pub n_decoded: i32,
    pub n_prompt: i32,
    pub timings: Option<Timings>,
}

#[derive(Debug, Clone)]
pub struct CompletionFinal {
    pub content: String,
    pub tokens: Vec<i32>,
    pub n_decoded: i32,
    pub n_prompt: i32,
    pub stop_type: StopType,
    pub stopping_word: String,
    pub truncated: bool,
    pub has_new_line: bool,
    pub tokens_cached: i32,
    pub prompt_text: String,
    pub timings: Timings,
}

pub fn legacy_chunk_json(chunk: &CompletionChunk) -> Value {
    let mut out = Map::new();
    out.insert("index".to_string(), json!(0));
    out.insert("content".to_string(), json!(chunk.content));
    out.insert("tokens".to_string(), json!([chunk.token]));
    out.insert("stop".to_string(), json!(false));
    out.insert("id_slot".to_string(), json!(0));
    out.insert("tokens_predicted".to_string(), json!(chunk.n_decoded));
    out.insert("tokens_evaluated".to_string(), json!(chunk.n_prompt));
    if let Some(timings) = &chunk.timings {
        out.insert("timings".to_string(), timings.to_json());
    }
    Value::Object(out)
}

pub fn legacy_final_json(
    final_result: &CompletionFinal,
    params: &CompletionParams,
    gen_settings: Value,
    stream: bool,
) -> Value {
    let content = if stream {
        String::new()
    } else {
        final_result.content.clone()
    };
    let tokens = if stream || !params.return_tokens {
        Vec::new()
    } else {
        final_result.tokens.clone()
    };

    let mut value = json!({
        "index": 0,
        "content": content,
        "tokens": tokens,
        "id_slot": 0,
        "stop": true,
        "model": params.model_name,
        "tokens_predicted": final_result.n_decoded,
        "tokens_evaluated": final_result.n_prompt,
        "prompt": final_result.prompt_text,
        "has_new_line": final_result.has_new_line,
        "truncated": final_result.truncated,
        "stop_type": final_result.stop_type.as_str(),
        "stopping_word": final_result.stopping_word,
        "tokens_cached": final_result.tokens_cached,
        "timings": final_result.timings.to_json(),
    });
    value["generation_settings"] = gen_settings;
    value
}

pub fn oai_chunk_json(chunk: &CompletionChunk, params: &CompletionParams, created: u64) -> Value {
    let mut out = Map::new();
    out.insert("id".to_string(), json!(params.cmpl_id));
    out.insert("object".to_string(), json!("text_completion"));
    out.insert("created".to_string(), json!(created));
    out.insert("model".to_string(), json!(params.model_name));
    out.insert("system_fingerprint".to_string(), json!(SYSTEM_FINGERPRINT));
    out.insert(
        "choices".to_string(),
        json!([{
            "text": chunk.content,
            "index": 0,
            "logprobs": Value::Null,
            "finish_reason": Value::Null,
        }]),
    );
    if let Some(timings) = &chunk.timings {
        out.insert("timings".to_string(), timings.to_json());
    }
    Value::Object(out)
}

pub fn oai_final_json(
    final_result: &CompletionFinal,
    params: &CompletionParams,
    created: u64,
) -> Value {
    let finish_reason = match final_result.stop_type {
        StopType::Word | StopType::Eos => "stop",
        StopType::None | StopType::Limit => "length",
    };
    let content = if params.stream {
        String::new()
    } else {
        final_result.content.clone()
    };

    json!({
        "id": params.cmpl_id,
        "object": "text_completion",
        "created": created,
        "model": params.model_name,
        "system_fingerprint": SYSTEM_FINGERPRINT,
        "choices": [{
            "text": content,
            "index": 0,
            "logprobs": Value::Null,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "completion_tokens": final_result.n_decoded,
            "prompt_tokens": final_result.n_prompt,
            "total_tokens": final_result.n_decoded + final_result.n_prompt,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
        "timings": final_result.timings.to_json(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matcher(stops: &[&str]) -> StopMatcher {
        StopMatcher::new(stops.iter().map(|s| (*s).to_string()).collect())
    }

    #[test]
    fn full_stop_match_spans_a_token_boundary() {
        let stops = matcher(&["STOP"]);
        assert_eq!(stops.scan("hello ST", 2, false), StopScan::None);
        assert_eq!(
            stops.scan("hello STOP", 2, false),
            StopScan::Full { pos: 6, word: 0 }
        );
    }

    #[test]
    fn full_stop_match_ignores_text_outside_the_window() {
        let stops = matcher(&["ab"]);
        assert_eq!(stops.scan("abXXXXXXXX", 1, false), StopScan::None);
        assert_eq!(
            stops.scan("abXXXXXXXX", 8, false),
            StopScan::Full { pos: 0, word: 0 }
        );
    }

    #[test]
    fn full_stop_match_at_the_window_edge_is_found() {
        let stops = matcher(&["end"]);
        assert_eq!(
            stops.scan("xxxendyy", 2, false),
            StopScan::Full { pos: 3, word: 0 }
        );
        assert_eq!(stops.scan("xxxendyy", 1, false), StopScan::None);
    }

    #[test]
    fn earliest_full_match_wins_across_multiple_stops() {
        let stops = matcher(&["world", "lo w"]);
        assert_eq!(
            stops.scan("hello world", 11, false),
            StopScan::Full { pos: 3, word: 1 }
        );
    }

    #[test]
    fn first_listed_stop_wins_on_a_tie() {
        let stops = matcher(&["abc", "ab"]);
        assert_eq!(
            stops.scan("zzabc", 5, false),
            StopScan::Full { pos: 2, word: 0 }
        );
    }

    #[test]
    fn partial_stop_is_held_back_then_diverges() {
        let stops = matcher(&["STOP"]);
        assert_eq!(
            stops.scan("hello ST", 2, true),
            StopScan::Partial { pos: 6 }
        );
        assert_eq!(stops.scan("hello STx", 1, true), StopScan::None);
    }

    #[test]
    fn partial_stop_prefers_the_longest_suffix() {
        let stops = matcher(&["abcd"]);
        assert_eq!(stops.scan("zzabc", 1, true), StopScan::Partial { pos: 2 });
    }

    #[test]
    fn partial_stop_is_not_reported_when_not_allowed() {
        let stops = matcher(&["STOP"]);
        assert_eq!(stops.scan("hello ST", 2, false), StopScan::None);
    }

    #[test]
    fn incremental_utf8_joins_a_split_three_byte_char() {
        let mut decoder = IncrementalUtf8::new();
        let bytes = "€".as_bytes();
        assert_eq!(decoder.push(&bytes[..2]), None);
        assert_eq!(decoder.push(&bytes[2..]), Some("€".to_string()));
        assert!(decoder.is_empty());
    }

    #[test]
    fn incremental_utf8_holds_the_whole_buffer_until_valid() {
        let mut decoder = IncrementalUtf8::new();
        let bytes = "a€".as_bytes();
        assert_eq!(decoder.push(&bytes[..2]), None);
        assert_eq!(decoder.push(&bytes[2..]), Some("a€".to_string()));
    }

    #[test]
    fn incremental_utf8_replaces_hard_invalid_bytes() {
        let mut decoder = IncrementalUtf8::new();
        assert_eq!(
            decoder.push(&[b'a', 0xff, b'b']),
            Some("a\u{fffd}b".to_string())
        );
        assert_eq!(decoder.push(b"ok"), Some("ok".to_string()));
    }

    #[test]
    fn incremental_utf8_passes_ascii_through() {
        let mut decoder = IncrementalUtf8::new();
        assert_eq!(decoder.push(b"hello"), Some("hello".to_string()));
        assert_eq!(decoder.push(b""), None);
    }

    fn defaults() -> SamplingDefaults {
        SamplingDefaults::default()
    }

    #[test]
    fn n_predict_alias_precedence() {
        let payload = json!({
            "prompt": "hi",
            "n_predict": 1,
            "max_completion_tokens": 2,
            "max_tokens": 3,
        });
        let params = CompletionParams::from_payload(&payload, &defaults(), 4096, "m").unwrap();
        assert_eq!(params.n_predict, 1);

        let payload = json!({"prompt": "hi", "max_completion_tokens": 2, "max_tokens": 3});
        let params = CompletionParams::from_payload(&payload, &defaults(), 4096, "m").unwrap();
        assert_eq!(params.n_predict, 2);

        let payload = json!({"prompt": "hi", "max_tokens": 3});
        let params = CompletionParams::from_payload(&payload, &defaults(), 4096, "m").unwrap();
        assert_eq!(params.n_predict, 3);

        let payload = json!({"prompt": "hi"});
        let params = CompletionParams::from_payload(&payload, &defaults(), 4096, "m").unwrap();
        assert_eq!(params.n_predict, -1);
    }

    #[test]
    fn prompt_string_and_token_array_are_accepted() {
        let params =
            CompletionParams::from_payload(&json!({"prompt": "hi"}), &defaults(), 4096, "m")
                .unwrap();
        assert_eq!(params.prompt, PromptInput::Text("hi".to_string()));

        let params =
            CompletionParams::from_payload(&json!({"prompt": [1, 2, 3]}), &defaults(), 4096, "m")
                .unwrap();
        assert_eq!(params.prompt, PromptInput::Tokens(vec![1, 2, 3]));
    }

    #[test]
    fn multiple_prompts_are_not_supported() {
        for prompt in [json!(["a", "b"]), json!([[1, 2], [3]]), json!([1, "a"])] {
            let err = CompletionParams::from_payload(
                &json!({ "prompt": prompt }),
                &defaults(),
                4096,
                "m",
            )
            .unwrap_err();
            assert_eq!(err.code, 501);
            assert_eq!(err.message, "multiple prompts not supported yet");
        }
    }

    #[test]
    fn missing_or_invalid_prompt_is_rejected() {
        let err = CompletionParams::from_payload(&json!({}), &defaults(), 4096, "m").unwrap_err();
        assert_eq!(err.code, 400);

        let err = CompletionParams::from_payload(&json!({"prompt": 7}), &defaults(), 4096, "m")
            .unwrap_err();
        assert_eq!(err.code, 400);

        let err = CompletionParams::from_payload(&json!({"prompt": []}), &defaults(), 4096, "m")
            .unwrap_err();
        assert_eq!(err.code, 400);
    }

    #[test]
    fn stop_filtering_drops_empty_strings() {
        let params = CompletionParams::from_payload(
            &json!({"prompt": "hi", "stop": ["", "a", "", "bb"]}),
            &defaults(),
            4096,
            "m",
        )
        .unwrap();
        assert_eq!(params.stop, vec!["a".to_string(), "bb".to_string()]);

        let err = CompletionParams::from_payload(
            &json!({"prompt": "hi", "stop": ["a", 1]}),
            &defaults(),
            4096,
            "m",
        )
        .unwrap_err();
        assert_eq!(err.code, 400);
    }

    #[test]
    fn model_name_falls_back_to_the_server_model_id() {
        let params = CompletionParams::from_payload(
            &json!({"prompt": "hi"}),
            &defaults(),
            4096,
            "server-id",
        )
        .unwrap();
        assert_eq!(params.model_name, "server-id");

        let params = CompletionParams::from_payload(
            &json!({"prompt": "hi", "model": "custom"}),
            &defaults(),
            4096,
            "server-id",
        )
        .unwrap();
        assert_eq!(params.model_name, "custom");
        assert!(params.cmpl_id.starts_with("chatcmpl-"));
    }

    #[test]
    fn timings_json_guards_division_by_zero() {
        let timings = Timings::default();
        let value = timings.to_json();
        assert_eq!(value["prompt_per_token_ms"], json!(0.0));
        assert_eq!(value["prompt_per_second"], json!(0.0));
        assert_eq!(value["predicted_per_token_ms"], json!(0.0));
        assert_eq!(value["predicted_per_second"], json!(0.0));
        assert_eq!(value["cache_n"], json!(0));

        let timings = Timings {
            prompt_n: 4,
            prompt_ms: 200.0,
            predicted_n: 2,
            predicted_ms: 100.0,
        };
        let value = timings.to_json();
        assert_eq!(value["prompt_per_token_ms"], json!(50.0));
        assert_eq!(value["prompt_per_second"], json!(20.0));
        assert_eq!(value["predicted_per_token_ms"], json!(50.0));
        assert_eq!(value["predicted_per_second"], json!(20.0));
    }

    fn sample_params() -> CompletionParams {
        CompletionParams::from_payload(
            &json!({"prompt": "hi", "return_tokens": true}),
            &defaults(),
            4096,
            "m",
        )
        .unwrap()
    }

    fn sample_final() -> CompletionFinal {
        CompletionFinal {
            content: "out".to_string(),
            tokens: vec![1, 2],
            n_decoded: 2,
            n_prompt: 3,
            stop_type: StopType::Eos,
            stopping_word: String::new(),
            truncated: false,
            has_new_line: false,
            tokens_cached: 1,
            prompt_text: "hi".to_string(),
            timings: Timings::default(),
        }
    }

    #[test]
    fn legacy_json_shapes_match_the_reference_server() {
        let chunk = CompletionChunk {
            content: "a".to_string(),
            token: 7,
            n_decoded: 1,
            n_prompt: 3,
            timings: None,
        };
        let value = legacy_chunk_json(&chunk);
        assert_eq!(value["tokens"], json!([7]));
        assert_eq!(value["stop"], json!(false));
        assert_eq!(value["tokens_predicted"], json!(1));
        assert!(value.get("timings").is_none());

        let params = sample_params();
        let value = legacy_final_json(&sample_final(), &params, json!({}), false);
        assert_eq!(value["content"], json!("out"));
        assert_eq!(value["tokens"], json!([1, 2]));
        assert_eq!(value["stop_type"], json!("eos"));
        assert_eq!(value["stop"], json!(true));

        let value = legacy_final_json(&sample_final(), &params, json!({}), true);
        assert_eq!(value["content"], json!(""));
        assert_eq!(value["tokens"], json!([]));
    }

    #[test]
    fn oai_json_shapes_match_the_reference_server() {
        let params = sample_params();
        let chunk = CompletionChunk {
            content: "a".to_string(),
            token: 7,
            n_decoded: 1,
            n_prompt: 3,
            timings: Some(Timings::default()),
        };
        let value = oai_chunk_json(&chunk, &params, 42);
        assert_eq!(value["object"], json!("text_completion"));
        assert_eq!(value["created"], json!(42));
        assert_eq!(value["choices"][0]["text"], json!("a"));
        assert_eq!(value["choices"][0]["finish_reason"], Value::Null);
        assert!(value.get("timings").is_some());

        let value = oai_final_json(&sample_final(), &params, 42);
        assert_eq!(value["choices"][0]["finish_reason"], json!("stop"));
        assert_eq!(value["usage"]["total_tokens"], json!(5));
        assert_eq!(
            value["usage"]["prompt_tokens_details"]["cached_tokens"],
            json!(0)
        );

        let mut limited = sample_final();
        limited.stop_type = StopType::Limit;
        let value = oai_final_json(&limited, &params, 42);
        assert_eq!(value["choices"][0]["finish_reason"], json!("length"));
    }

    #[test]
    fn oai_stream_final_carries_finish_reason_and_usage_without_content() {
        let params = CompletionParams::from_payload(
            &json!({"prompt": "hi", "stream": true}),
            &defaults(),
            4096,
            "m",
        )
        .unwrap();
        let value = oai_final_json(&sample_final(), &params, 42);
        assert_eq!(value["choices"][0]["text"], json!(""));
        assert_eq!(value["choices"][0]["finish_reason"], json!("stop"));
        assert_eq!(value["usage"]["completion_tokens"], json!(2));
        assert!(value.get("timings").is_some());
    }
}
