use crate::config::ServerConfig;
use crate::engine::task::TaskError;
use llama_cpp::model::{AddBos, LlamaModel};
use llama_cpp::sampling::LlamaSampler;
use llama_cpp::token::logit_bias::LlamaLogitBias;
use llama_cpp::token::LlamaToken;
use serde_json::{json, Map, Value};

/// `LLAMA_DEFAULT_SEED`: the seed value that makes llama.cpp pick a random one.
pub const DEFAULT_SEED: u32 = u32::MAX;

const DEFAULT_DRY_SEQUENCE_BREAKERS: [&str; 4] = ["\n", ":", "\"", "*"];

const DEFAULT_SAMPLER_NAMES: [&str; 9] = [
    "penalties",
    "dry",
    "top_n_sigma",
    "top_k",
    "typ_p",
    "top_p",
    "min_p",
    "xtc",
    "temperature",
];

const MIROSTAT_M: i32 = 100;

#[derive(Debug, Clone, PartialEq)]
pub enum LogitBiasEntry {
    Token(i32, f32),
    Text(String, f32),
}

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub seed: u32,
    pub min_keep: usize,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub typical_p: f32,
    pub temperature: f32,
    pub dynatemp_range: f32,
    pub dynatemp_exponent: f32,
    pub penalty_last_n: i32,
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: i32,
    pub dry_penalty_last_n: i32,
    pub dry_sequence_breakers: Vec<String>,
    pub xtc_probability: f32,
    pub xtc_threshold: f32,
    pub mirostat: i32,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
    pub ignore_eos: bool,
    pub logit_bias: Vec<LogitBiasEntry>,
}

#[derive(Debug, Clone)]
pub struct SamplingDefaults {
    pub seed: u32,
    pub min_keep: usize,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub typical_p: f32,
    pub temperature: f32,
    pub dynatemp_range: f32,
    pub dynatemp_exponent: f32,
    pub penalty_last_n: i32,
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: i32,
    pub dry_penalty_last_n: i32,
    pub dry_sequence_breakers: Vec<String>,
    pub xtc_probability: f32,
    pub xtc_threshold: f32,
    pub mirostat: i32,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
    pub ignore_eos: bool,
    pub n_predict: i32,
}

impl Default for SamplingDefaults {
    fn default() -> Self {
        Self {
            seed: DEFAULT_SEED,
            min_keep: 0,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            typical_p: 1.00,
            temperature: 0.80,
            dynatemp_range: 0.00,
            dynatemp_exponent: 1.00,
            penalty_last_n: 64,
            penalty_repeat: 1.00,
            penalty_freq: 0.00,
            penalty_present: 0.00,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: -1,
            dry_sequence_breakers: DEFAULT_DRY_SEQUENCE_BREAKERS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            xtc_probability: 0.00,
            xtc_threshold: 0.10,
            mirostat: 0,
            mirostat_tau: 5.00,
            mirostat_eta: 0.10,
            ignore_eos: false,
            n_predict: -1,
        }
    }
}

impl SamplingDefaults {
    pub fn from_config(config: &ServerConfig) -> Self {
        Self {
            top_k: config.top_k,
            top_p: config.top_p,
            min_p: config.min_p,
            temperature: config.temperature,
            n_predict: config.n_predict,
            ..Self::default()
        }
    }
}

impl SamplingParams {
    pub fn from_payload(
        payload: &Value,
        defaults: &SamplingDefaults,
        n_ctx: i32,
    ) -> Result<Self, TaskError> {
        let mut params = Self {
            seed: json_seed(payload, "seed", defaults.seed),
            min_keep: json_usize(payload, "min_keep", defaults.min_keep),
            top_k: json_i32(payload, "top_k", defaults.top_k),
            top_p: json_f32(payload, "top_p", defaults.top_p),
            min_p: json_f32(payload, "min_p", defaults.min_p),
            typical_p: json_f32(payload, "typical_p", defaults.typical_p),
            temperature: json_f32(payload, "temperature", defaults.temperature),
            dynatemp_range: json_f32(payload, "dynatemp_range", defaults.dynatemp_range),
            dynatemp_exponent: json_f32(payload, "dynatemp_exponent", defaults.dynatemp_exponent),
            penalty_last_n: json_i32(payload, "repeat_last_n", defaults.penalty_last_n),
            penalty_repeat: json_f32(payload, "repeat_penalty", defaults.penalty_repeat),
            penalty_freq: json_f32(payload, "frequency_penalty", defaults.penalty_freq),
            penalty_present: json_f32(payload, "presence_penalty", defaults.penalty_present),
            dry_multiplier: json_f32(payload, "dry_multiplier", defaults.dry_multiplier),
            dry_base: json_f32(payload, "dry_base", defaults.dry_base),
            dry_allowed_length: json_i32(
                payload,
                "dry_allowed_length",
                defaults.dry_allowed_length,
            ),
            dry_penalty_last_n: json_i32(
                payload,
                "dry_penalty_last_n",
                defaults.dry_penalty_last_n,
            ),
            dry_sequence_breakers: defaults.dry_sequence_breakers.clone(),
            xtc_probability: json_f32(payload, "xtc_probability", defaults.xtc_probability),
            xtc_threshold: json_f32(payload, "xtc_threshold", defaults.xtc_threshold),
            mirostat: json_i32(payload, "mirostat", defaults.mirostat),
            mirostat_tau: json_f32(payload, "mirostat_tau", defaults.mirostat_tau),
            mirostat_eta: json_f32(payload, "mirostat_eta", defaults.mirostat_eta),
            ignore_eos: json_bool(payload, "ignore_eos", defaults.ignore_eos),
            logit_bias: parse_logit_bias(payload.get("logit_bias")),
        };

        if params.penalty_last_n < -1 {
            return Err(TaskError::invalid_request("repeat_last_n must be >= -1"));
        }
        if params.dry_penalty_last_n < -1 {
            return Err(TaskError::invalid_request(
                "dry_penalty_last_n must be >= -1",
            ));
        }
        if params.penalty_last_n == -1 {
            params.penalty_last_n = n_ctx;
        }
        if params.dry_penalty_last_n == -1 {
            params.dry_penalty_last_n = n_ctx;
        }
        if params.dry_base < 1.0 {
            params.dry_base = defaults.dry_base;
        }

        if payload.get("dry_sequence_breakers").is_some() {
            let breakers = parse_string_array(&payload["dry_sequence_breakers"]);
            if breakers.is_empty() {
                return Err(TaskError::invalid_request(
                    "dry_sequence_breakers must be a non-empty array of strings",
                ));
            }
            params.dry_sequence_breakers = breakers;
        }

        Ok(params)
    }

    pub fn build_sampler(&self, model: &LlamaModel, eog_bias: &[LlamaLogitBias]) -> LlamaSampler {
        let n_vocab = model.n_vocab();
        let mut biases = Vec::new();
        for entry in &self.logit_bias {
            match entry {
                LogitBiasEntry::Token(token, bias) => {
                    if *token >= 0 && *token < n_vocab {
                        biases.push(LlamaLogitBias::new(LlamaToken::new(*token), *bias));
                    }
                }
                LogitBiasEntry::Text(text, bias) => {
                    if let Ok(tokens) = model.str_to_token(text, AddBos::Never) {
                        biases.extend(tokens.into_iter().map(|t| LlamaLogitBias::new(t, *bias)));
                    }
                }
            }
        }
        if self.ignore_eos {
            biases.extend_from_slice(eog_bias);
        }

        let mut chain = Vec::new();
        if !biases.is_empty() {
            chain.push(LlamaSampler::logit_bias(n_vocab, &biases));
        }

        match self.mirostat {
            1 => {
                chain.push(LlamaSampler::temp(self.temperature));
                chain.push(LlamaSampler::mirostat(
                    n_vocab,
                    self.seed,
                    self.mirostat_tau,
                    self.mirostat_eta,
                    MIROSTAT_M,
                ));
            }
            2 => {
                chain.push(LlamaSampler::temp(self.temperature));
                chain.push(LlamaSampler::mirostat_v2(
                    self.seed,
                    self.mirostat_tau,
                    self.mirostat_eta,
                ));
            }
            _ => {
                let breakers = self
                    .dry_sequence_breakers
                    .iter()
                    .filter(|s| !s.as_bytes().contains(&0))
                    .cloned()
                    .collect::<Vec<_>>();
                chain.push(LlamaSampler::penalties(
                    self.penalty_last_n,
                    self.penalty_repeat,
                    self.penalty_freq,
                    self.penalty_present,
                ));
                chain.push(LlamaSampler::dry(
                    model,
                    self.dry_multiplier,
                    self.dry_base,
                    self.dry_allowed_length,
                    self.dry_penalty_last_n,
                    breakers,
                ));
                chain.push(LlamaSampler::top_k(self.top_k));
                chain.push(LlamaSampler::typical(self.typical_p, self.min_keep));
                chain.push(LlamaSampler::top_p(self.top_p, self.min_keep));
                chain.push(LlamaSampler::min_p(self.min_p, self.min_keep));
                chain.push(LlamaSampler::xtc(
                    self.xtc_probability,
                    self.xtc_threshold,
                    self.min_keep,
                    self.seed,
                ));
                chain.push(LlamaSampler::temp_ext(
                    self.temperature,
                    self.dynatemp_range,
                    self.dynatemp_exponent,
                ));
                chain.push(LlamaSampler::dist(self.seed));
            }
        }

        LlamaSampler::chain_simple(chain)
    }

    pub fn generation_settings_json(
        &self,
        n_predict: i32,
        _n_ctx: i32,
        stop: &[String],
        stream: bool,
        _model_name: &str,
    ) -> Value {
        let logit_bias = self
            .logit_bias
            .iter()
            .map(|entry| match entry {
                LogitBiasEntry::Token(token, bias) => json!({"bias": bias, "token": token}),
                LogitBiasEntry::Text(text, bias) => json!({"bias": bias, "text": text}),
            })
            .collect::<Vec<_>>();

        let parts = [
            json!({
                "seed": self.seed,
                "temperature": self.temperature,
                "dynatemp_range": self.dynatemp_range,
                "dynatemp_exponent": self.dynatemp_exponent,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "min_p": self.min_p,
                "top_n_sigma": -1.0,
                "xtc_probability": self.xtc_probability,
                "xtc_threshold": self.xtc_threshold,
                "typical_p": self.typical_p,
            }),
            json!({
                "repeat_last_n": self.penalty_last_n,
                "repeat_penalty": self.penalty_repeat,
                "presence_penalty": self.penalty_present,
                "frequency_penalty": self.penalty_freq,
                "dry_multiplier": self.dry_multiplier,
                "dry_base": self.dry_base,
                "dry_allowed_length": self.dry_allowed_length,
                "dry_penalty_last_n": self.dry_penalty_last_n,
                "dry_sequence_breakers": self.dry_sequence_breakers,
                "mirostat": self.mirostat,
                "mirostat_tau": self.mirostat_tau,
                "mirostat_eta": self.mirostat_eta,
            }),
            json!({
                "stop": stop,
                "max_tokens": n_predict,
                "n_predict": n_predict,
                "n_keep": 0,
                "n_discard": 0,
                "ignore_eos": self.ignore_eos,
                "stream": stream,
                "logit_bias": logit_bias,
                "n_probs": 0,
                "min_keep": self.min_keep,
                "grammar": "",
                "grammar_lazy": false,
            }),
            json!({
                "grammar_triggers": [],
                "preserved_tokens": [],
                "chat_format": "Content-only",
                "reasoning_format": "none",
                "reasoning_in_content": false,
                "generation_prompt": "",
                "samplers": DEFAULT_SAMPLER_NAMES,
                "speculative.type": "none",
                "timings_per_token": false,
                "post_sampling_probs": false,
                "backend_sampling": false,
                "lora": [],
            }),
        ];

        let mut settings = Map::new();
        for part in parts {
            if let Value::Object(map) = part {
                settings.extend(map);
            }
        }
        Value::Object(settings)
    }
}

fn json_field<'a>(payload: &'a Value, key: &str) -> Option<&'a Value> {
    match payload.get(key) {
        None | Some(Value::Null) => None,
        Some(value) => Some(value),
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn json_number_i64(payload: &Value, key: &str) -> Option<i64> {
    let value = json_field(payload, key)?;
    if !value.is_number() {
        return None;
    }
    if let Some(v) = value.as_i64() {
        return Some(v);
    }
    if let Some(v) = value.as_u64() {
        return Some(v as i64);
    }
    value.as_f64().map(|v| v as i64)
}

#[allow(clippy::cast_possible_truncation)]
pub fn json_f32(payload: &Value, key: &str, default: f32) -> f32 {
    json_field(payload, key)
        .and_then(Value::as_f64)
        .map_or(default, |value| value as f32)
}

#[allow(clippy::cast_possible_truncation)]
pub fn json_i32(payload: &Value, key: &str, default: i32) -> i32 {
    json_number_i64(payload, key).map_or(default, |value| value as i32)
}

pub fn json_i64(payload: &Value, key: &str, default: i64) -> i64 {
    json_number_i64(payload, key).unwrap_or(default)
}

pub fn json_bool(payload: &Value, key: &str, default: bool) -> bool {
    json_field(payload, key)
        .and_then(Value::as_bool)
        .unwrap_or(default)
}

pub fn json_usize(payload: &Value, key: &str, default: usize) -> usize {
    json_number_i64(payload, key).map_or(default, |value| usize::try_from(value).unwrap_or(0))
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn json_seed(payload: &Value, key: &str, default: u32) -> u32 {
    json_number_i64(payload, key).map_or(default, |value| value as u32)
}

fn parse_string_array(value: &Value) -> Vec<String> {
    let Some(items) = value.as_array() else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        match item.as_str() {
            Some(text) => out.push(text.to_string()),
            None => return Vec::new(),
        }
    }
    out
}

#[allow(clippy::cast_possible_truncation)]
fn parse_bias_value(value: &Value) -> Option<f32> {
    match value {
        Value::Number(_) => value.as_f64().map(|v| v as f32),
        Value::Bool(false) => Some(f32::NEG_INFINITY),
        _ => None,
    }
}

#[allow(clippy::cast_possible_truncation)]
fn parse_logit_bias(value: Option<&Value>) -> Vec<LogitBiasEntry> {
    let mut out = Vec::new();
    match value {
        Some(Value::Array(items)) => {
            for item in items {
                let Some(pair) = item.as_array() else {
                    continue;
                };
                if pair.len() != 2 {
                    continue;
                }
                let Some(bias) = parse_bias_value(&pair[1]) else {
                    continue;
                };
                if let Some(token) = pair[0].as_i64() {
                    out.push(LogitBiasEntry::Token(token as i32, bias));
                } else if let Some(text) = pair[0].as_str() {
                    out.push(LogitBiasEntry::Text(text.to_string(), bias));
                }
            }
        }
        Some(Value::Object(entries)) => {
            for (key, value) in entries {
                let Some(bias) = parse_bias_value(value) else {
                    continue;
                };
                match strtol_exact(key) {
                    Some(token) => out.push(LogitBiasEntry::Token(token, bias)),
                    None => out.push(LogitBiasEntry::Text(key.clone(), bias)),
                }
            }
        }
        _ => {}
    }
    out
}

#[allow(clippy::cast_possible_truncation)]
fn strtol_exact(key: &str) -> Option<i32> {
    let bytes = key.as_bytes();
    if bytes.is_empty() {
        return Some(0);
    }
    let mut i = 0;
    while i < bytes.len() && matches!(bytes[i], b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r') {
        i += 1;
    }
    let negative = match bytes.get(i) {
        Some(b'-') => {
            i += 1;
            true
        }
        Some(b'+') => {
            i += 1;
            false
        }
        _ => false,
    };
    let digits_start = i;
    let mut value: i64 = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        value = value
            .saturating_mul(10)
            .saturating_add(i64::from(bytes[i] - b'0'));
        i += 1;
    }
    if i == digits_start || i != bytes.len() {
        return None;
    }
    if negative {
        value = -value;
    }
    Some(value as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn defaults() -> SamplingDefaults {
        SamplingDefaults::default()
    }

    #[test]
    fn defaults_are_applied_when_fields_are_absent() {
        let params = SamplingParams::from_payload(&json!({}), &defaults(), 4096).unwrap();
        assert_eq!(params.top_k, 40);
        assert!((params.top_p - 0.95).abs() < f32::EPSILON);
        assert!((params.temperature - 0.80).abs() < f32::EPSILON);
        assert!((params.dry_base - 1.75).abs() < f32::EPSILON);
        assert_eq!(params.penalty_last_n, 64);
        assert_eq!(params.dry_penalty_last_n, 4096);
        assert_eq!(params.seed, DEFAULT_SEED);
        assert_eq!(
            params.dry_sequence_breakers,
            vec![
                "\n".to_string(),
                ":".to_string(),
                "\"".to_string(),
                "*".to_string()
            ]
        );
        assert!(params.logit_bias.is_empty());
        assert!(!params.ignore_eos);
    }

    #[test]
    fn repeat_last_n_below_minus_one_is_rejected() {
        let err = SamplingParams::from_payload(&json!({"repeat_last_n": -2}), &defaults(), 4096)
            .unwrap_err();
        assert_eq!(err.code, 400);
        assert_eq!(err.message, "repeat_last_n must be >= -1");
    }

    #[test]
    fn dry_penalty_last_n_below_minus_one_is_rejected() {
        let err =
            SamplingParams::from_payload(&json!({"dry_penalty_last_n": -3}), &defaults(), 4096)
                .unwrap_err();
        assert_eq!(err.code, 400);
    }

    #[test]
    fn repeat_last_n_minus_one_becomes_n_ctx() {
        let params =
            SamplingParams::from_payload(&json!({"repeat_last_n": -1}), &defaults(), 2048).unwrap();
        assert_eq!(params.penalty_last_n, 2048);
    }

    #[test]
    fn dry_base_below_one_resets_to_default() {
        let params =
            SamplingParams::from_payload(&json!({"dry_base": 0.5}), &defaults(), 4096).unwrap();
        assert!((params.dry_base - 1.75).abs() < f32::EPSILON);
    }

    #[test]
    fn negative_seed_maps_to_default_seed() {
        let params = SamplingParams::from_payload(&json!({"seed": -1}), &defaults(), 4096).unwrap();
        assert_eq!(params.seed, u32::MAX);
    }

    #[test]
    fn type_mismatched_fields_fall_back_to_defaults() {
        let params = SamplingParams::from_payload(
            &json!({"top_k": "many", "temperature": null, "ignore_eos": 1}),
            &defaults(),
            4096,
        )
        .unwrap();
        assert_eq!(params.top_k, 40);
        assert!((params.temperature - 0.80).abs() < f32::EPSILON);
        assert!(!params.ignore_eos);
    }

    #[test]
    fn logit_bias_array_form_is_parsed() {
        let params = SamplingParams::from_payload(
            &json!({"logit_bias": [[15, -1.0], ["hello", 2.0], [7, false], [3, "x"], "bad"]}),
            &defaults(),
            4096,
        )
        .unwrap();
        assert_eq!(
            params.logit_bias,
            vec![
                LogitBiasEntry::Token(15, -1.0),
                LogitBiasEntry::Text("hello".to_string(), 2.0),
                LogitBiasEntry::Token(7, f32::NEG_INFINITY),
            ]
        );
    }

    #[test]
    fn logit_bias_object_form_is_parsed() {
        let params = SamplingParams::from_payload(
            &json!({"logit_bias": {"15": -1.0, "hello": 2.0, "7": false, "12abc": 1.0}}),
            &defaults(),
            4096,
        )
        .unwrap();
        let mut entries = params.logit_bias;
        entries.sort_by_key(|entry| match entry {
            LogitBiasEntry::Token(token, _) => format!("0{token:08}"),
            LogitBiasEntry::Text(text, _) => format!("1{text}"),
        });
        assert_eq!(
            entries,
            vec![
                LogitBiasEntry::Token(7, f32::NEG_INFINITY),
                LogitBiasEntry::Token(15, -1.0),
                LogitBiasEntry::Text("12abc".to_string(), 1.0),
                LogitBiasEntry::Text("hello".to_string(), 2.0),
            ]
        );
    }

    #[test]
    fn ignore_eos_is_parsed() {
        let params =
            SamplingParams::from_payload(&json!({"ignore_eos": true}), &defaults(), 4096).unwrap();
        assert!(params.ignore_eos);
    }

    #[test]
    fn dry_sequence_breakers_must_be_a_non_empty_string_array() {
        let params = SamplingParams::from_payload(
            &json!({"dry_sequence_breakers": ["a", "bb"]}),
            &defaults(),
            4096,
        )
        .unwrap();
        assert_eq!(
            params.dry_sequence_breakers,
            vec!["a".to_string(), "bb".to_string()]
        );

        for bad in [json!([]), json!("nope"), json!([1, 2]), Value::Null] {
            let err = SamplingParams::from_payload(
                &json!({"dry_sequence_breakers": bad}),
                &defaults(),
                4096,
            )
            .unwrap_err();
            assert_eq!(err.code, 400);
        }
    }

    #[test]
    fn strtol_exact_matches_c_semantics() {
        assert_eq!(strtol_exact("15"), Some(15));
        assert_eq!(strtol_exact(" -3"), Some(-3));
        assert_eq!(strtol_exact("+9"), Some(9));
        assert_eq!(strtol_exact(""), Some(0));
        assert_eq!(strtol_exact("12abc"), None);
        assert_eq!(strtol_exact("abc"), None);
        assert_eq!(strtol_exact("1 "), None);
    }

    #[test]
    fn generation_settings_json_has_the_full_key_set() {
        let params = SamplingParams::from_payload(&json!({}), &defaults(), 4096).unwrap();
        let settings =
            params.generation_settings_json(128, 4096, &["stop".to_string()], true, "model");
        let object = settings.as_object().unwrap();
        for key in [
            "seed",
            "temperature",
            "dynatemp_range",
            "dynatemp_exponent",
            "top_k",
            "top_p",
            "min_p",
            "top_n_sigma",
            "xtc_probability",
            "xtc_threshold",
            "typical_p",
            "repeat_last_n",
            "repeat_penalty",
            "presence_penalty",
            "frequency_penalty",
            "dry_multiplier",
            "dry_base",
            "dry_allowed_length",
            "dry_penalty_last_n",
            "dry_sequence_breakers",
            "mirostat",
            "mirostat_tau",
            "mirostat_eta",
            "stop",
            "max_tokens",
            "n_predict",
            "n_keep",
            "n_discard",
            "ignore_eos",
            "stream",
            "logit_bias",
            "n_probs",
            "min_keep",
            "grammar",
            "grammar_lazy",
            "grammar_triggers",
            "preserved_tokens",
            "chat_format",
            "reasoning_format",
            "reasoning_in_content",
            "generation_prompt",
            "samplers",
            "speculative.type",
            "timings_per_token",
            "post_sampling_probs",
            "backend_sampling",
            "lora",
        ] {
            assert!(object.contains_key(key), "missing key: {key}");
        }
        assert_eq!(settings["n_predict"], json!(128));
        assert_eq!(settings["stop"], json!(["stop"]));
        assert_eq!(settings["stream"], json!(true));
        assert_eq!(settings["samplers"], json!(DEFAULT_SAMPLER_NAMES));
    }
}
