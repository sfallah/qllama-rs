use crate::config::ServerConfig;
use anyhow::{anyhow, Context};
use llama_cpp::context::params::LlamaContextParams;
use llama_cpp::context::LlamaContext;
use llama_cpp::llama_backend::LlamaBackend;
use llama_cpp::llama_batch::LlamaBatch;
use llama_cpp::model::params::LlamaModelParams;
use llama_cpp::model::{AddBos, LlamaModel};
use llama_cpp::openai::OpenAIChatTemplateParams;
use llama_cpp::sampling::LlamaSampler;
use llama_cpp::token::LlamaToken;
use serde_json::{json, Value};
use std::num::NonZeroU32;

pub struct EngineRuntime {
    _backend: LlamaBackend,
    model: &'static LlamaModel,
    ctx: LlamaContext<'static>,
    default_n_predict: i32,
    default_temperature: f32,
    default_top_k: i32,
    default_top_p: f32,
    default_min_p: f32,
}

impl EngineRuntime {
    pub fn new(config: &ServerConfig) -> anyhow::Result<Self> {
        let backend = LlamaBackend::init().context("failed to initialize llama backend")?;
        let model = LlamaModel::load_from_file(
            &backend,
            &config.model_path,
            &LlamaModelParams::default(),
        )
            .with_context(|| format!("failed to load model from {}", config.model_path.display()))?;
        let model_ref: &'static LlamaModel = Box::leak(Box::new(model));
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(config.n_ctx))
            .with_n_batch(config.n_batch)
            .with_n_ubatch(config.n_ubatch);
        let ctx = model_ref
            .new_context(&backend, ctx_params)
            .context("failed to create llama context")?;

        Ok(Self {
            _backend: backend,
            model: model_ref,
            ctx,
            default_n_predict: config.n_predict,
            default_temperature: config.temperature,
            default_top_k: config.top_k,
            default_top_p: config.top_p,
            default_min_p: config.min_p,
        })
    }

    pub fn tokenize(&self, payload: &Value) -> anyhow::Result<Value> {
        let content = payload
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("missing string field: content"))?;
        let with_pieces = payload
            .get("with_pieces")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let tokens = self.model.str_to_token(content, AddBos::Never)?;
        if with_pieces {
            let mut decoder = encoding_rs::UTF_8.new_decoder();
            let pieces = tokens
                .iter()
                .map(|token| {
                    let piece = self
                        .model
                        .token_to_piece(*token, &mut decoder, true, None)
                        .unwrap_or_default();
                    json!({"id": token.0, "piece": piece})
                })
                .collect::<Vec<_>>();
            Ok(json!({ "tokens": pieces }))
        } else {
            Ok(json!({ "tokens": tokens.iter().map(|t| t.0).collect::<Vec<_>>() }))
        }
    }

    pub fn detokenize(&self, payload: &Value) -> anyhow::Result<Value> {
        let token_values = payload
            .get("tokens")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("missing array field: tokens"))?;
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut out = String::new();
        for token in token_values {
            let token_id = token
                .as_i64()
                .ok_or_else(|| anyhow!("tokens entries must be integers"))?;
            let piece = self.model.token_to_piece(
                LlamaToken::new(i32::try_from(token_id)?),
                &mut decoder,
                true,
                None,
            )?;
            out.push_str(&piece);
        }
        Ok(json!({ "content": out }))
    }

    pub fn apply_template(&self, payload: &Value) -> anyhow::Result<Value> {
        let messages = payload
            .get("messages")
            .ok_or_else(|| anyhow!("missing field: messages"))?;
        let tools_json = payload.get("tools").map(Value::to_string);
        let json_schema = payload.get("json_schema").map(Value::to_string);
        let chat_template_kwargs = payload.get("chat_template_kwargs").map(Value::to_string);
        let params = OpenAIChatTemplateParams {
            messages_json: &messages.to_string(),
            tools_json: tools_json.as_deref(),
            tool_choice: payload.get("tool_choice").and_then(Value::as_str),
            json_schema: json_schema.as_deref(),
            grammar: payload.get("grammar").and_then(Value::as_str),
            reasoning_format: payload.get("reasoning_format").and_then(Value::as_str),
            chat_template_kwargs: chat_template_kwargs.as_deref(),
            add_generation_prompt: payload
                .get("add_generation_prompt")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            use_jinja: payload.get("use_jinja").and_then(Value::as_bool).unwrap_or(true),
            parallel_tool_calls: payload
                .get("parallel_tool_calls")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            enable_thinking: payload
                .get("enable_thinking")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            add_bos: payload.get("add_bos").and_then(Value::as_bool).unwrap_or(true),
            add_eos: payload.get("add_eos").and_then(Value::as_bool).unwrap_or(false),
            parse_tool_calls: payload
                .get("parse_tool_calls")
                .and_then(Value::as_bool)
                .unwrap_or(true),
        };
        let tmpl = self
            .model
            .chat_template(payload.get("chat_template").and_then(Value::as_str))?;
        let prompt = self
            .model
            .apply_chat_template_oaicompat(&tmpl, &params)?;
        Ok(json!({ "prompt": prompt.prompt }))
    }

    pub fn completion(&mut self, payload: &Value) -> anyhow::Result<Value> {
        let prompt = payload
            .get("prompt")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("missing string field: prompt"))?;
        let n_predict = payload
            .get("n_predict")
            .or_else(|| payload.get("max_tokens"))
            .and_then(Value::as_i64)
            .unwrap_or(i64::from(self.default_n_predict));
        let temperature = payload
            .get("temperature")
            .and_then(Value::as_f64)
            .map(|v| v as f32)
            .unwrap_or(self.default_temperature);
        let top_k = payload
            .get("top_k")
            .and_then(Value::as_i64)
            .map(i32::try_from)
            .transpose()?
            .unwrap_or(self.default_top_k);
        let top_p = payload
            .get("top_p")
            .and_then(Value::as_f64)
            .map(|v| v as f32)
            .unwrap_or(self.default_top_p);
        let min_p = payload
            .get("min_p")
            .and_then(Value::as_f64)
            .map(|v| v as f32)
            .unwrap_or(self.default_min_p);
        let seed = payload
            .get("seed")
            .and_then(Value::as_u64)
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(1234);

        let mut tokens = self.model.str_to_token(prompt, AddBos::Always)?;
        if tokens.is_empty() {
            return Err(anyhow!("prompt tokenization returned empty input"));
        }

        self.ctx.clear_kv_cache();
        let mut batch = LlamaBatch::new(self.ctx.n_batch() as usize, 1);
        let last = i32::try_from(tokens.len())? - 1;
        for (i, token) in (0_i32..).zip(tokens.iter().copied()) {
            batch.add(token, i, &[0], i == last)?;
        }
        self.ctx.decode(&mut batch)?;

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_k(top_k),
            LlamaSampler::top_p(top_p, 1),
            LlamaSampler::min_p(min_p, 1),
            LlamaSampler::temp(temperature),
            LlamaSampler::dist(seed),
        ]);
        sampler.accept_many(tokens.iter());

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut generated = String::new();
        let mut generated_tokens = Vec::<i32>::new();
        let mut n_cur = batch.n_tokens();
        let n_limit = i32::try_from(n_predict.max(0))?;

        for _ in 0..n_limit {
            let token = sampler.sample(&self.ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if self.model.is_eog_token(token) {
                break;
            }
            generated.push_str(
                &self
                    .model
                    .token_to_piece(token, &mut decoder, true, None)
                    .unwrap_or_default(),
            );
            generated_tokens.push(token.0);
            tokens.push(token);

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;
            self.ctx.decode(&mut batch)?;
        }

        Ok(json!({
            "content": generated,
            "tokens": generated_tokens,
            "prompt": prompt,
            "model": "local",
            "stop": true
        }))
    }

    pub fn model_n_ctx(&self) -> u32 {
        self.ctx.n_ctx()
    }
}
