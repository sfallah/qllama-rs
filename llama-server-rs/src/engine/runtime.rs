use crate::config::ServerConfig;
use crate::engine::completion::{
    CompletionChunk, CompletionFinal, CompletionParams, IncrementalUtf8, PromptInput, StopMatcher,
    StopScan, StopType, Timings,
};
use crate::engine::sampler::SamplingDefaults;
use crate::engine::task::TaskError;
use anyhow::{anyhow, Context};
use llama_cpp::context::params::LlamaContextParams;
use llama_cpp::context::LlamaContext;
use llama_cpp::llama_backend::LlamaBackend;
use llama_cpp::llama_batch::LlamaBatch;
use llama_cpp::model::params::LlamaModelParams;
use llama_cpp::model::{AddBos, LlamaModel};
use llama_cpp::openai::OpenAIChatTemplateParams;
use llama_cpp::token::logit_bias::LlamaLogitBias;
use llama_cpp::token::LlamaToken;
use llama_cpp::TokenToStringError;
use serde_json::{json, Value};
use std::num::NonZeroU32;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

/// Buffer size `token_to_piece` starts with before retrying with the exact size.
const PIECE_BUFFER_SIZE: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitOutcome {
    Continue,
    ReceiverGone,
}

/// Per-request generation state. Everything the token loop mutates lives here so
/// that moving generation onto a slot pool only changes who owns it.
#[derive(Debug, Default)]
struct GenerationState {
    content: String,
    unsent: String,
    tokens: Vec<i32>,
    last_token: i32,
    n_decoded: i32,
    n_past: i32,
    stop_type: StopType,
    stopping_word: String,
    truncated: bool,
    has_new_line: bool,
}

impl GenerationState {
    /// Mirrors the stop-word window of `server_slot::find_stopping_strings`: a full
    /// match truncates the buffer, a partial match holds its tail back for the next
    /// token, and anything else is released in full.
    fn take_sent(&mut self, piece_len: usize, stops: &StopMatcher) -> String {
        let mut send_len = self.unsent.len();
        let mut discard_rest = false;
        match stops.scan(&self.unsent, piece_len, true) {
            StopScan::Full { pos, word } => {
                send_len = pos;
                self.stop_type = StopType::Word;
                self.stopping_word.clone_from(&stops.stops()[word]);
                discard_rest = true;
            }
            StopScan::Partial { pos } => send_len = pos,
            StopScan::None => {}
        }
        let sent = take_prefix(&mut self.unsent, send_len);
        if discard_rest {
            self.unsent.clear();
        }
        sent
    }
}

fn take_prefix(buffer: &mut String, len: usize) -> String {
    let mut len = len.min(buffer.len());
    while len > 0 && !buffer.is_char_boundary(len) {
        len -= 1;
    }
    let rest = buffer.split_off(len);
    std::mem::replace(buffer, rest)
}

fn elapsed_ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}

/// A partial stop that never completed is released once generation ends; on a
/// word stop the tail after the match has already been dropped.
fn flush_held_text(
    state: &mut GenerationState,
    params: &CompletionParams,
    emit: &mut dyn FnMut(CompletionChunk) -> EmitOutcome,
    n_prompt: i32,
    prompt_ms: f64,
    predict_start: Instant,
) -> Result<(), TaskError> {
    if state.stop_type == StopType::Word || state.unsent.is_empty() {
        return Ok(());
    }
    let held = std::mem::take(&mut state.unsent);
    state.content.push_str(&held);
    if held.contains('\n') {
        state.has_new_line = true;
    }
    if params.stream {
        let chunk = CompletionChunk {
            content: held,
            token: state.last_token,
            n_decoded: state.n_decoded,
            n_prompt,
            timings: Some(Timings {
                prompt_n: n_prompt,
                prompt_ms,
                predicted_n: state.n_decoded,
                predicted_ms: elapsed_ms(predict_start),
            }),
        };
        if emit(chunk) == EmitOutcome::ReceiverGone {
            return Err(TaskError::server("client disconnected"));
        }
    }
    Ok(())
}

pub struct EngineRuntime {
    _backend: LlamaBackend,
    model: &'static LlamaModel,
    ctx: LlamaContext<'static>,
    eog_bias: Vec<LlamaLogitBias>,
    pub defaults: SamplingDefaults,
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

        let eog_bias = (0..model_ref.n_vocab())
            .map(LlamaToken::new)
            .filter(|token| model_ref.is_eog_token(*token))
            .map(|token| LlamaLogitBias::new(token, f32::NEG_INFINITY))
            .collect();

        Ok(Self {
            _backend: backend,
            model: model_ref,
            ctx,
            eog_bias,
            defaults: SamplingDefaults::from_config(config),
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

    #[allow(clippy::too_many_lines)]
    pub fn completion(
        &mut self,
        params: &CompletionParams,
        cancel: &CancellationToken,
        emit: &mut dyn FnMut(CompletionChunk) -> EmitOutcome,
    ) -> Result<CompletionFinal, TaskError> {
        let n_ctx = i32::try_from(self.ctx.n_ctx()).unwrap_or(i32::MAX);
        let prompt_tokens = self.prompt_tokens(&params.prompt)?;
        if prompt_tokens.is_empty() {
            return Err(TaskError::invalid_request("\"prompt\" must not be empty"));
        }
        let n_prompt = i32::try_from(prompt_tokens.len()).unwrap_or(i32::MAX);
        if n_prompt >= n_ctx {
            return Err(TaskError::invalid_request(
                "the request exceeds the available context size",
            ));
        }

        let (mut sample_idx, prompt_ms) = self.decode_prompt(&prompt_tokens)?;
        let mut sampler = params.sampling.build_sampler(self.model, &self.eog_bias);
        sampler.accept_many(prompt_tokens.iter());

        let stops = StopMatcher::new(params.stop.clone());
        let mut utf8 = IncrementalUtf8::new();
        let mut batch = LlamaBatch::new(1, 1);
        let mut state = GenerationState {
            n_past: n_prompt,
            ..GenerationState::default()
        };
        let predict_start = Instant::now();

        loop {
            if cancel.is_cancelled() {
                return Err(TaskError::server("request cancelled"));
            }

            let token = sampler.sample(&self.ctx, sample_idx);
            sampler.accept(token);
            state.n_decoded += 1;
            state.last_token = token.0;
            if params.return_tokens {
                state.tokens.push(token.0);
            }
            if self.model.is_eog_token(token) {
                state.stop_type = StopType::Eos;
                break;
            }

            let piece = utf8.push(&self.piece_bytes(token));
            let mut has_next = true;
            let mut sent_new_line = false;
            if utf8.is_empty() {
                let piece_len = piece.as_ref().map_or(0, String::len);
                if let Some(piece) = piece {
                    state.unsent.push_str(&piece);
                }
                let sent = state.take_sent(piece_len, &stops);
                state.content.push_str(&sent);
                sent_new_line = sent.contains('\n');
                has_next = state.stop_type == StopType::None;
                if params.stream {
                    let chunk = CompletionChunk {
                        content: sent,
                        token: token.0,
                        n_decoded: state.n_decoded,
                        n_prompt,
                        timings: (params.timings_per_token || !has_next).then(|| Timings {
                            prompt_n: n_prompt,
                            prompt_ms,
                            predicted_n: state.n_decoded,
                            predicted_ms: elapsed_ms(predict_start),
                        }),
                    };
                    if emit(chunk) == EmitOutcome::ReceiverGone {
                        return Err(TaskError::server("client disconnected"));
                    }
                }
            }

            if state.n_past + 1 >= n_ctx {
                state.truncated = true;
                state.stop_type = StopType::Limit;
                has_next = false;
            }
            if has_next && params.n_predict >= 0 && state.n_decoded >= params.n_predict {
                state.stop_type = StopType::Limit;
                has_next = false;
            }
            if sent_new_line {
                state.has_new_line = true;
                let budget = u128::try_from(params.t_max_predict_ms).unwrap_or(u128::MAX);
                if params.t_max_predict_ms > 0 && predict_start.elapsed().as_millis() > budget {
                    state.stop_type = StopType::Limit;
                    has_next = false;
                }
            }
            if !has_next {
                break;
            }

            batch.clear();
            batch
                .add(token, state.n_past, &[0], true)
                .map_err(|err| TaskError::server(format!("failed to build batch: {err}")))?;
            state.n_past += 1;
            self.ctx
                .decode(&mut batch)
                .map_err(|err| TaskError::server(format!("failed to decode token: {err}")))?;
            sample_idx = 0;
        }

        flush_held_text(&mut state, params, emit, n_prompt, prompt_ms, predict_start)?;

        Ok(CompletionFinal {
            content: state.content,
            tokens: state.tokens,
            n_decoded: state.n_decoded,
            n_prompt,
            stop_type: state.stop_type,
            stopping_word: state.stopping_word,
            truncated: state.truncated,
            has_new_line: state.has_new_line,
            tokens_cached: state.n_past,
            prompt_text: match &params.prompt {
                PromptInput::Text(text) => text.clone(),
                PromptInput::Tokens(_) => self.detokenize_tokens(&prompt_tokens),
            },
            timings: Timings {
                prompt_n: n_prompt,
                prompt_ms,
                predicted_n: state.n_decoded,
                predicted_ms: elapsed_ms(predict_start),
            },
        })
    }

    fn prompt_tokens(&self, prompt: &PromptInput) -> Result<Vec<LlamaToken>, TaskError> {
        match prompt {
            PromptInput::Text(text) => self.model.str_to_token(text, AddBos::Always).map_err(|err| {
                TaskError::invalid_request(format!("failed to tokenize prompt: {err}"))
            }),
            PromptInput::Tokens(ids) => {
                let n_vocab = self.model.n_vocab();
                ids.iter()
                    .map(|id| {
                        if *id >= 0 && *id < n_vocab {
                            Ok(LlamaToken::new(*id))
                        } else {
                            Err(TaskError::invalid_request(format!(
                                "invalid token id in \"prompt\": {id}"
                            )))
                        }
                    })
                    .collect()
            }
        }
    }

    fn decode_prompt(&mut self, tokens: &[LlamaToken]) -> Result<(i32, f64), TaskError> {
        let started = Instant::now();
        self.ctx.clear_kv_cache();

        let n_batch = usize::try_from(self.ctx.n_batch()).unwrap_or(1).max(1);
        let mut batch = LlamaBatch::new(n_batch, 1);
        let last = i32::try_from(tokens.len()).unwrap_or(i32::MAX) - 1;
        let mut pos = 0_i32;
        for chunk in tokens.chunks(n_batch) {
            batch.clear();
            for token in chunk {
                batch
                    .add(*token, pos, &[0], pos == last)
                    .map_err(|err| TaskError::server(format!("failed to build batch: {err}")))?;
                pos += 1;
            }
            self.ctx
                .decode(&mut batch)
                .map_err(|err| TaskError::server(format!("failed to decode prompt: {err}")))?;
        }

        Ok((batch.n_tokens() - 1, elapsed_ms(started)))
    }

    fn piece_bytes(&self, token: LlamaToken) -> Vec<u8> {
        match self
            .model
            .token_to_piece_bytes(token, PIECE_BUFFER_SIZE, false, None)
        {
            Ok(bytes) => bytes,
            Err(TokenToStringError::InsufficientBufferSpace(size)) => self
                .model
                .token_to_piece_bytes(token, usize::try_from(-size).unwrap_or(0), false, None)
                .unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    }

    fn detokenize_tokens(&self, tokens: &[LlamaToken]) -> String {
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut out = String::new();
        for token in tokens {
            if let Ok(piece) = self.model.token_to_piece(*token, &mut decoder, true, None) {
                out.push_str(&piece);
            }
        }
        out
    }

    pub fn model_n_ctx(&self) -> u32 {
        self.ctx.n_ctx()
    }
}
