# llama-server-rs ↔ llama.cpp/tools/server Parity Plan

## 1. Overview

**Scope.** Bring `llama-server-rs` (Axum-based Rust crate at `llama-server-rs`) to full functional parity with the C++ reference server at `llama-cpp-sys/llama.cpp/tools/server`. Parity targets the public REST + SSE surface (routes registered in `server.cpp:172–224`) plus all engine-side features those routes depend on (slot scheduler, continuous batching, prompt cache, speculative decoding, multimodal `mtmd`, LoRA hot-swap, grammar/JSON-Schema/llguidance, samplers, Jinja chat templates, tool-call parsing).

**Non-goals.** Web UI assets (`public/`, `webui/`, `index.html`, `bundle.js/css`), CORS proxy + built-in tools (experimental, behind `--webui-mcp-proxy` / `--server-tools` flags). Router (multi-model proxy: `models_routes`) is **deferred to a later phase**, but the `/models/load`, `/models/unload` handlers and `/v1/models` listing must remain wired.

**Reference versions.** Vendored llama.cpp under `llama-cpp-sys/llama.cpp` (path: `tools/server`). Rust bindings: sibling crate `llama-cpp` (already exposes sampling, batching, kv_cache, sessions, mtmd, openai chat parsing, grammar, llguidance — see `llama-cpp/src/lib.rs`, `sampling.rs`, `openai.rs`, `mtmd.rs`).

---

## 2. Architecture Comparison

| Layer | C++ server | Current Rust (`llama-server-rs`) | Gap |
|---|---|---|---|
| HTTP transport | `cpp-httplib` via `server-http.cpp`, route registration in `server.cpp:172`. | Axum 0.7 router in `http/mod.rs`. | Routes wired but every handler is a JSON-passthrough stub. |
| Auth | `is_public_endpoint` list + `Authorization`/`X-Api-Key` (`server-http.cpp`). | `http/auth.rs` — equivalent logic. | Missing public-list parity for `/v1/health`, slots-disable cases, and constant-time compare. |
| Server context | `server_context` (`server-context.cpp`) owns model, contexts, slots, queue, prompt cache, mtmd, speculative draft model, LoRA stack. | `engine::Engine` wraps a queue + dummy slot loop; **no model is ever loaded**. | The entire inference layer is a stub. `llama_backend`, `LlamaModel`, `LlamaContext` are not instantiated. |
| Task queue | `server_queue` (priority + deferred lanes, callbacks `on_new_task`/`on_update_slots`). | `engine/queue.rs` — simple MPSC + deferred VecDeque. | No priority, no completion routing, no cancellation propagation, only one broadcast per task. |
| Slot scheduler | `server_slot` array; `update_slots()` continuous batcher; KV-prefix reuse; per-slot `server_prompt` cache; speculative draft model. | `engine/slot.rs` — just `{id, active_task_id, phase}`. | No batching, no KV reuse, no decoding loop. |
| Sampling | `common_sampler` chain (penalties, dry, top_k, typ_p, top_p, min_p, xtc, mirostat, temperature, grammar, logit-bias). | `engine/sampler.rs` — stores 4 fields, never used. | Need full chain construction from request payload using `LlamaSampler::*` from `llama-cpp/src/sampling.rs`. |
| Chat templating | `chat.cpp` + Jinja via `minja`; OAI-compat parser `chat_parse_state_oaicompat`. | `engine/chat.rs` — returns `payload["messages"]` only. | Use `OpenAIChatTemplateParams` + `ChatParseStateOaicompat` from `llama-cpp::openai`. |
| Multimodal | `mtmd_helper`, `server_tokens` mixed text+image chunks. | `engine/mtmd.rs` — returns raw payload. | Wire `llama-cpp::mtmd` into prompt-build pipeline. |
| Prompt cache | `server_prompt_cache` LRU on disk + in-memory KV state via `llama_state_seq_*`. | `engine/prompt_cache.rs` — generic byte LRU, unused. | Bind to `LlamaContext::state_seq_save/load`. |
| Streaming | httplib chunked-encoding generator (`server_res_generator`); SSE + NDJSON. | `http/sse.rs` — generic `chunk`/`done`/`error` events. | Per-endpoint encoders (NDJSON for `/completion`, OAI SSE `data: {...}\n\n…[DONE]`, Anthropic event stream). |
| Metrics | Prometheus text in `routes.get_metrics`: `n_prompt_tokens_total`, `n_predicted_tokens_total`, `kv_cache_*`, `requests_*`, slot-busy gauge. | `http/handlers/metrics.rs` — 4 hand-formatted counters. | Build `prometheus::Registry` covering full C++ metric set. |
| CLI / config | `common_params` ≈ 250 flags. | `cli.rs` — ~7 flags. | Massive expansion (see §6). |

---

## 3. Endpoint Parity Matrix

Status legend: ✅ done, 🟡 wired but stub, 🟥 missing/handler-only.
Priority: P0 = critical (Phase 1–2), P1 = important (3–4), P2 = nice-to-have (5–7).

| # | Method | Path | C++ source | Rust file | C++ | Rust | Gap | Pri |
|---|---|---|---|---|---|---|---|---|
| 1 | GET | `/health` | `server.cpp:172` | `handlers/health.rs` | ✅ | 🟡 | Queue/slot status, `error.code=503` while loading. | P0 |
| 2 | GET | `/v1/health` | `server.cpp:173` | same | ✅ | 🟡 | Alias of #1. | P0 |
| 3 | GET | `/metrics` | `server.cpp:174` | `handlers/metrics.rs` | ✅ | 🟡 | Replace with full Prometheus registry; gated by `--metrics`. | P1 |
| 4 | GET | `/props` | `server.cpp:175` | `handlers/props.rs` | ✅ | 🟡 | Return `default_generation_settings`, `chat_template`, `chat_template_caps`, `modalities`, `media_marker`, `build_info`, `is_sleeping`, `total_slots`, `model_path`. | P1 |
| 5 | POST | `/props` | `server.cpp:176` | same | ✅ | 🟥 | Mutates global defaults, gated by `--props`. | P2 |
| 6 | GET | `/models` | `server.cpp:177` | `handlers/models.rs` | ✅ | 🟡 | Add `meta`, `created`, `permission`. | P1 |
| 7 | GET | `/v1/models` | `server.cpp:178` | same | ✅ | 🟡 | Alias of #6. | P1 |
| 8 | POST | `/completion` | `server.cpp:179` | `handlers/completions.rs::post_completion` | ✅ | 🟥 | Full legacy schema (§4.A); NDJSON streaming. | P0 |
| 9 | POST | `/completions` | `server.cpp:180` | same | ✅ | 🟥 | Alias of #8. | P0 |
| 10 | POST | `/v1/completions` | `server.cpp:181` | `post_completions_oai` | ✅ | 🟥 | OAI text-completion shape; SSE + `[DONE]`. | P0 |
| 11 | POST | `/chat/completions` | `server.cpp:182` | `handlers/chat.rs` | ✅ | 🟥 | OAI chat, Jinja tmpl, tools, response_format, streaming deltas, usage, multimodal `image_url`, finish_reason. | P0 |
| 12 | POST | `/v1/chat/completions` | `server.cpp:183` | same | ✅ | 🟥 | Alias of #11. | P0 |
| 13 | POST | `/v1/responses` | `server.cpp:184` | `post_responses` | ✅ | 🟥 | OAI Responses API. | P2 |
| 14 | POST | `/responses` | `server.cpp:185` | same | ✅ | 🟥 | Alias of #13. | P2 |
| 15 | POST | `/v1/audio/transcriptions` | `server.cpp:186` | `handlers/transcriptions.rs` | ✅ | 🟥 | multipart/form-data; needs Whisper / audio-mtmd. | P2 |
| 16 | POST | `/audio/transcriptions` | `server.cpp:187` | same | ✅ | 🟥 | Alias of #15. | P2 |
| 17 | POST | `/v1/messages` | `server.cpp:188` | `post_anthropic_messages` | ✅ | 🟥 | Anthropic Messages API + named SSE events. | P2 |
| 18 | POST | `/v1/messages/count_tokens` | `server.cpp:189` | `post_anthropic_count_tokens` | ✅ | 🟥 | `{input_tokens}` only. | P2 |
| 19 | POST | `/infill` | `server.cpp:190` | `handlers/infill.rs` | ✅ | 🟥 | FIM payload + same gen pipeline. | P1 |
| 20 | POST | `/embedding` | `server.cpp:191` | `handlers/embeddings.rs` | ✅ | 🟥 | Legacy: `[{index, embedding:[[…tokens]]}]`; supports `--pooling none`. | P1 |
| 21 | POST | `/embeddings` | `server.cpp:192` | same | ✅ | 🟥 | Same as #20. | P1 |
| 22 | POST | `/v1/embeddings` | `server.cpp:193` | `post_embeddings_oai` (shared) | ✅ | 🟥 | OAI shape; `encoding_format: float|base64`. | P1 |
| 23 | POST | `/rerank` | `server.cpp:194` | `handlers/rerank.rs` | ✅ | 🟥 | Native + TEI body. | P1 |
| 24 | POST | `/reranking` | `server.cpp:195` | same | ✅ | 🟥 | Alias. | P1 |
| 25 | POST | `/v1/rerank` | `server.cpp:196` | same | ✅ | 🟥 | Alias. | P1 |
| 26 | POST | `/v1/reranking` | `server.cpp:197` | same | ✅ | 🟥 | Alias. | P1 |
| 27 | POST | `/tokenize` | `server.cpp:198` | `handlers/tokenize.rs` | ✅ | 🟥 | `{content, add_special?, with_pieces?}` → tokens. | P0 |
| 28 | POST | `/detokenize` | `server.cpp:199` | same | ✅ | 🟥 | `{tokens}` → `{content}`. | P0 |
| 29 | POST | `/apply-template` | `server.cpp:200` | `handlers/apply_template.rs` | ✅ | 🟥 | Render Jinja chat template. | P0 |
| 30 | GET | `/lora-adapters` | `server.cpp:202` | `handlers/lora.rs::get_lora` | ✅ | 🟥 | List `[{id, path, scale}]`. | P1 |
| 31 | POST | `/lora-adapters` | `server.cpp:203` | `post_lora` | ✅ | 🟥 | Hot-swap scales. | P1 |
| 32 | GET | `/slots` | `server.cpp:205` | `handlers/slots.rs::get_slots` | ✅ | 🟥 | Per-slot snapshot; `?fail_on_no_slot=1`; `--no-slots`. | P1 |
| 33 | POST | `/slots/:id_slot` | `server.cpp:206` | `post_slot` | ✅ | 🟥 | `?action=save\|restore\|erase`; gated by `--slot-save-path`. | P2 |
| 34 | POST | `/models/load` | `server.cpp:168` | `handlers/router.rs` | ✅ (router) | 🟥 | Defer until router phase. | P3 |
| 35 | POST | `/models/unload` | `server.cpp:169` | same | ✅ (router) | 🟥 | Defer. | P3 |
| — | OPTIONS | `/*` | n/a | `http/mod.rs:74` | n/a | ✅ | CORS preflight. | — |

Excluded: `/cors-proxy`, `/tools` (experimental), all static asset routes (`/`, `/bundle.js`, `/bundle.css`).

---

## 4. Per-Endpoint Specs

References (server `README.md`): `POST /completion` (L382–572), `POST /v1/chat/completions` (L1194+), `GET /props` (L727), `POST /embeddings` (L831), `POST /reranking` (L660), `POST /infill` (L693), `GET /slots` (L871), `POST /slots/{id_slot}?action=…` (L1034/1054/1074), `GET/POST /lora-adapters` (L1085/1112), `GET /metrics` (L1019).

### A. `/completion` (legacy native)

- **Request fields:** `prompt` (string|tokens|array), `temperature`, `dynatemp_range`, `dynatemp_exponent`, `top_k`, `top_p`, `min_p`, `n_predict`/`max_tokens`, `n_indent`, `n_keep`, `stream`, `stop`, `typical_p`, `repeat_penalty`, `repeat_last_n`, `presence_penalty`, `frequency_penalty`, `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_penalty_last_n`, `dry_sequence_breakers`, `xtc_probability`, `xtc_threshold`, `mirostat`, `mirostat_tau`, `mirostat_eta`, `grammar`, `grammar_lazy`, `grammar_triggers`, `preserved_tokens`, `json_schema`, `seed`, `ignore_eos`, `logit_bias`, `n_probs`, `min_keep`, `t_max_predict_ms`, `image_data`, `id_slot`, `cache_prompt`, `return_tokens`, `samplers`, `timings_per_token`, `post_sampling_probs`, `response_fields`, `lora`.
- **Response (non-stream):** `{content, tokens?, generation_settings, prompt, has_new_line, truncated, stop_type, stopping_word, tokens_cached, timings:{...}, index, completion_probabilities?, model, id_slot, …}`.
- **Streaming:** newline-delimited JSON; final chunk `stop:true`.
- **Errors:** 400 invalid, 503 no slot, 504 cancelled; envelope `{error:{message,type,code}}`.

### B. `/v1/completions`

- OAI subset of A: `model`, `prompt`, `max_tokens`, `temperature`, `top_p`, `n`, `stream`, `stop`, `logit_bias`, `seed`, `frequency_penalty`, `presence_penalty`, `logprobs`, `echo`, `suffix`, `user`.
- **Response:** `{id, object:"text_completion", created, model, choices:[{text, index, logprobs, finish_reason}], usage, timings?}`.
- **Streaming:** `data: {…}\n\n` + `data: [DONE]\n\n`.

### C. `/chat/completions` & `/v1/chat/completions`

- All A fields plus `messages` (multimodal parts: `{type:"text"}`, `{type:"image_url", image_url:{url}}`), `tools`, `tool_choice`, `response_format` (`{type:"json_object"|"json_schema", schema?}`), `chat_template`, `chat_template_kwargs`, `reasoning_format`, `parallel_tool_calls`, `enable_thinking`, `parse_tool_calls`, `add_generation_prompt`, `model`, `function_call` (legacy).
- **Response:** `{id:"chatcmpl-…", object:"chat.completion", created, model, choices:[{index, message:{role,content,tool_calls?,reasoning_content?}, finish_reason, logprobs?}], usage:{prompt_tokens,completion_tokens,total_tokens}, timings, system_fingerprint}`.
- **Streaming:** `chat.completion.chunk` deltas; final chunk with `finish_reason` then `data: [DONE]`.
- Use `OpenAIChatTemplateParams` for prompt rendering; `ChatParseStateOaicompat::update` for tool/text deltas.

### D. `/v1/embeddings` (OAI)

- **Request:** `{model, input: str|str[]|int[]|int[][], encoding_format: "float"|"base64", dimensions?, user?}`.
- **Response:** `{object:"list", data:[{object:"embedding", index, embedding}], model, usage:{prompt_tokens,total_tokens}}`.
- `encoding_format=base64` → little-endian `f32` buffer base64-encoded.

### E. `/embedding`, `/embeddings` (legacy)

- **Request:** `{content|input, …}`.
- **Response:** `[{index, embedding:[[…token0…],…]}]` (token-level when `--pooling none`).

### F. `/rerank` family

- **Native:** `{query, documents, top_n?}`. **TEI:** `{query, texts, top_n?, truncate?, raw_scores?, return_text?}`.
- **Response:** `{model, results:[{index, relevance_score, document?:str}], usage}`.

### G. `/infill`

- **Request:** `input_prefix`, `input_suffix`, `input_extra:[{filename,text}]`, `prompt`, plus all A sampling fields.
- **Response:** same shape as A.

### H. `/tokenize`, `/detokenize`

- `{content, add_special?, with_pieces?}` → `{tokens:[int] | [{id, piece}]}`.
- `{tokens:[int]}` → `{content: str}`.

### I. `/apply-template`

- `{messages, tools?, tool_choice?, chat_template?, chat_template_kwargs?, add_generation_prompt?, use_jinja?}` → `{prompt: str}`.

### J. `/props` (GET) / `/props` (POST)

- See server README L734–812. POST gated by `--props`; body keys mutate `params_base` (currently only sampling defaults).

### K. `/slots` (GET) / `/slots/:id_slot` (POST)

- **GET:** array of `{id, id_task, n_ctx, speculative, is_processing, params, prompt, next_token}`.
- **POST `?action=save|restore|erase`:** body `{filename}` (relative to `--slot-save-path`); response `{id_slot, filename, n_saved, n_restored?, n_written/n_read, timings}`.

### L. `/lora-adapters` GET/POST

- **GET:** `[{id, path, scale}]`.
- **POST:** `[{id, scale}]` → `204` (or updated list); applies via `llama_set_adapter_lora`.

### M. `/metrics`

- Prometheus exposition. Required series: `llamacpp:prompt_tokens_total`, `llamacpp:tokens_predicted_total`, `llamacpp:prompt_tokens_seconds`, `llamacpp:predicted_tokens_seconds`, `llamacpp:kv_cache_usage_ratio`, `llamacpp:kv_cache_tokens`, `llamacpp:requests_processing`, `llamacpp:requests_deferred`, `llamacpp:n_decode_total`, `llamacpp:n_busy_slots_per_decode`, plus `build_info` gauge.

### N. `/health`

- 200 `{status:"ok"}` when ready; 503 `{error:{code:503,message:"Loading model",type:"unavailable_error"}}` while loading; 500 on init error.

### O. `/v1/messages` & `/v1/messages/count_tokens` (Anthropic)

- Map to internal chat task; emit Anthropic event names; `count_tokens` only renders template + tokenizes.

### P. `/v1/audio/transcriptions`

- multipart: `file`, `model`, `language?`, `prompt?`, `temperature?`, `response_format ∈ {json,text,srt,verbose_json,vtt}`. Requires audio-capable mtmd — defer behind feature flag.

---

## 5. Engine / Runtime Gaps

1. **Model lifecycle (P0).** Load `LlamaModel` + `LlamaContext` per slot at boot; expose via `AppState`. Add `--n-ctx`, `--n-batch`, `--n-ubatch`, `--n-parallel`, `--ctx-shift`, `--flash-attn`, `--rope-*`, `--cache-type-k/v`, `--mlock`, `--no-mmap`, `--n-gpu-layers`, `--main-gpu`, `--tensor-split`, `--numa`. New files: `engine/model.rs`, `engine/runtime.rs`. Replace stub `engine/loop.rs`.
2. **Slot scheduler & continuous batching (P0).** Replace `engine/slot.rs` with full state (sequence id, tokens-cached, params, kv-prefix, draft stats, prompt-cache handle). Implement `update_slots()` analogue: pack pending slot prompts into one `LlamaBatch`, decode, distribute logits per-seq, advance samplers. Mirror `server-context.cpp::server_context::update_slots`.
3. **Queue overhaul (P1).** Task → result-channel map (replace single broadcast). Priorities (control vs gen vs embed), cancellation propagation, `defer` on full slots, `fail_on_no_slot`.
4. **Sampler chain (P0).** New `engine/sampler/build.rs`: penalties → dry → top_k → typ_p → top_p → min_p → xtc → temp → grammar/llguidance → dist. Wire `samplers` ordering, `mirostat`, `logit_bias`, `seed`. Bindings in `llama-cpp/src/sampling.rs`.
5. **Grammar / JSON-Schema / llguidance (P1).** `grammar` → `LlamaSampler::grammar`; `json_schema` / `response_format.schema` → `json_schema_to_grammar`; `--grammar-llguidance` → `LlamaSampler::llguidance`. Lazy grammar with `grammar_triggers`/`preserved_tokens`.
6. **Prompt cache (P1).** Replace `engine/prompt_cache.rs` with two tiers:
   - **In-memory KV reuse:** common-prefix detection vs cached tokens; `llama_kv_cache_seq_rm` for divergent suffix.
   - **Disk persistence:** `LlamaContext::state_seq_save/load` under `--slot-save-path`.
7. **Multimodal pipeline (P2).** Wire `llama-cpp::mtmd` (`MtmdContext`, `MtmdBitmap`, `MtmdInputChunks`). Decode `messages[*].content[*].image_url.url` (data: URI or HTTP), build chunks, replace media markers, decode via `mtmd_helper_eval`. Gate on `--mmproj`.
8. **Speculative decoding (P2).** Optional draft model via `--model-draft`, `--ctx-size-draft`. Per-task params `speculative.{n_min,n_max,p_min}`. Requires new FFI shims in `llama-cpp-sys` (`common_speculative_*` is not in public `llama.h`).
9. **LoRA hot-swap (P1).** Use existing `LlamaLoraAdapter*` types; verify/expose `LlamaContext::set_adapter_lora(&adapter, scale)` and `clear_adapter_lora`.
10. **Embeddings & pooling (P1).** Honor `--pooling {none,mean,cls,last,rank}`; for `none` return per-token vectors via `embeddings_seq_ith`/`embeddings_ith`. Normalize unless `--embd-normalize 0`.
11. **Reranking (P1).** Run with `--pooling rank`; concatenate `query` + `BOS` + `doc` per pair; emit single logit.
12. **Infill (P1).** Inject FIM tokens (`fim_pre/_mid/_suf/_rep/_sep`) via tokenizer special-token API; reuse generation path.
13. **Chat templating + tool calls (P0).** `OpenAIChatTemplateParams` for rendering; `ChatParseStateOaicompat::update` for OAI deltas (text vs `tool_calls`). Honor `--jinja`, `--chat-template`, `--chat-template-file`, `--reasoning-format`.
14. **Cancellation (P1).** Bridge Axum disconnected future → `TaskHandle::cancel`. Slot worker checks `cancel.is_cancelled()` between tokens; reports `stopped_cancel=true`.
15. **Logit bias (P0).** Parse `[[token,bias]|[string,bias]|[token,false]]`, expand strings via tokenizer, build `LlamaSampler::logit_bias`.
16. **DRY / XTC / Mirostat / penalties (P0).** All exposed in bindings — wire from request payload.

---

## 6. Cross-Cutting Concerns

1. **Auth (`http/auth.rs`).** Add `/v1/health` to public list (currently present). Allow repeated `--api-key` and `--api-key-file`. Use constant-time compare. Public list mirrors `server-http.cpp` (`/health`, `/v1/health`, `/models`, `/v1/models`, web assets).
2. **CORS (`http/cors.rs`).** OK; add `Access-Control-Expose-Headers: x-request-id`.
3. **SSE / streaming (`http/sse.rs`).** Replace generic encoder with per-endpoint encoders:
   - OAI SSE (`data: <json>\n\n` + `[DONE]`),
   - Anthropic event stream (named events),
   - Legacy NDJSON for `/completion` & `/infill` (raw chunked, not SSE).
4. **Error format (`http/error.rs`).** Extend `AppError`: `NotFound(404)`, `ServiceUnavailable(503)`, `Conflict(409)`, `PayloadTooLarge(413)`, `RequestTimeout(504)`. Streaming errors emit final `error` event with `{error:{message,type,code}}` then close.
5. **Request cancellation.** Bridge axum/hyper response-drop → `cancel.cancel()`.
6. **Metrics.** `prometheus::Registry`; counters/gauges for queue depth, slots busy, prompt/predicted tokens, kv usage; expose at `/metrics`.
7. **Logging.** `tracing-subscriber` with EnvFilter, JSON option (`--log-json`), per-request span (`x-request-id`, slot id, task id, kind, latency). Add `tower-http::trace::TraceLayer`.
8. **CLI parity (`cli.rs`, `config.rs`).** Expand to llama.cpp `common_params`. Groups:
   - **Common:** `--model/-m`, `--alias`, `--hf-repo`, `--hf-file`, `--lora`, `--lora-scaled`, `--mmproj`, `--threads/-t`, `--threads-batch`, `--ctx-size/-c`, `--predict/-n`, `--batch-size/-b`, `--ubatch-size/-ub`, `--keep`, `--rope-scaling`, `--rope-freq-{base,scale}`, `--yarn-*`, `--cache-type-{k,v}`, `--n-gpu-layers/-ngl`, `--split-mode`, `--main-gpu`, `--tensor-split`, `--no-mmap`, `--mlock`, `--numa`, `--check-tensors`, `--flash-attn`.
   - **Sampling defaults** (mirrored into `default_generation_settings`).
   - **Server-specific:** `--host`, `--port`, `--path`, `--api-prefix`, `--api-key`, `--api-key-file`, `--ssl-key-file`, `--ssl-cert-file`, `--threads-http`, `--cache-reuse`, `--metrics`, `--props`, `--slots`/`--no-slots`, `--slot-save-path`, `--chat-template`, `--chat-template-file`, `--jinja`, `--reasoning-format`, `--no-context-shift`, `--parallel`, `--cont-batching`/`--no-cont-batching`, `--embeddings`, `--reranking`, `--pooling`, `--embd-normalize`, `--embd-output-format`, `--draft-max`, `--draft-min`, `--draft-p-min`, `--model-draft`, `--ctx-size-draft`, `--device-draft`, `--n-gpu-layers-draft`, `--lora-init-without-apply`, `--log-disable`, `--log-file`, `--verbose`, `--system-prompt-file`, `--mmproj-url`, `--special`.
9. **Dependencies (`Cargo.toml`).** Use already-present `prometheus`; add `mime`/`mime_guess` and `axum::extract::Multipart` (axum-extra `multipart` feature) for transcriptions; promote `reqwest` to runtime deps for `image_url` fetch; add `image` crate (or rely on mtmd helpers); `pin-project-lite` for cancellation streams; enable `tokio` `signal` feature.
10. **TLS.** Honor `--ssl-key-file` / `--ssl-cert-file` via existing `axum-server`+`rustls`.

---

## 7. Phased Roadmap

**Phase 1 — Engine foundation & native completion (P0).**
- Load model + N contexts; rewrite `engine/loop.rs` into a real slot scheduler.
- Full sampler chain; `/completion`, `/completions`, `/tokenize`, `/detokenize`, `/apply-template`.
- Real `/health`, `/v1/models`, `/props` GET, basic `/metrics`.
- Cancellation hook + per-task channels.

**Phase 2 — OAI compat + streaming (P0).**
- `/v1/completions`, `/chat/completions`, `/v1/chat/completions` with Jinja, OAI SSE, tool-call parsing.
- JSON-Schema/grammar (`response_format`, `grammar`, `json_schema`).
- Anthropic `/v1/messages` + `count_tokens`.

**Phase 3 — Embeddings, rerank, infill (P1).**
- `/v1/embeddings`, `/embeddings`, `/embedding` with pooling + base64.
- `/rerank` family (native + TEI).
- `/infill` with FIM tokens.

**Phase 4 — Multimodal (P1).**
- Wire `llama-cpp::mtmd` for chat `image_url`; remote URL fetch + base64 decode.
- Update `/props.modalities`, `media_marker`.

**Phase 5 — Slots, LoRA, prompt cache persistence (P1).**
- `/slots` GET (with `fail_on_no_slot`), `/slots/:id` save/restore/erase via session save/load.
- `/lora-adapters` GET/POST hot-swap.
- Disk prompt-cache LRU.

**Phase 6 — Speculative decoding + advanced sampling (P2).**
- Add `speculative` FFI shims; second draft model lifecycle.
- DRY/XTC/mirostat polish; `samplers` ordering; logit-bias by string.
- `/v1/responses` API.

**Phase 7 — Hardening (P1).**
- Full Prometheus set; slot-busy histogram; build_info.
- Constant-time auth, request-size limits, timeouts.
- Audio transcriptions behind feature flag.
- Soak / fuzz / golden-fixture comparisons against C++ server.

---

## 8. Testing Strategy

1. **Unit tests** alongside each engine module — sampler builder, chat-template golden outputs, prompt-cache prefix detection, logit-bias parsing.
2. **Integration tests** (`tests/`) — split `smoke_fixture.rs` into `tests/completion_native.rs`, `tests/chat_oai.rs`, `tests/embeddings.rs`, `tests/rerank.rs`, `tests/slots.rs`, `tests/lora.rs`. Use existing tiny GGUF fixtures (`llama-cpp/src/gguf/ggml-vocab-bert-bge.gguf`) for tokenizer-only tests; add a tiny generation-capable fixture for completions. Stream tests via `eventsource-stream` + `reqwest`.
3. **Golden compatibility tests** (feature `golden`) — spin up vendored C++ server, replay request fixtures, compare JSON shape (numeric tolerance) and SSE event sequence.
4. **Property tests** (proptest) — tokenize/detokenize round-trip; JSON-schema → grammar → sampler-accepts-only-valid output.
5. **Smoke fixture** — boot server with vocab-only fixture; assert `/health`, `/v1/models`, `/props`, auth rejection, CORS preflight, OPTIONS, error envelope shape.
6. **Load / soak** — `wrk`/`oha` script in `tests/scripts/` exercising parallel `/v1/chat/completions` streaming; assert no slot leaks (slot count returns to idle, queue depth zero).

---

## 9. Open Questions / Risks

1. **Router mode.** Defer to Phase 8; have `/models/load`/`/models/unload` return 501 until then.
2. **Audio transcriptions.** Whisper not exposed by `llama-cpp-sys`. Options: (A) 501 stub, (B) new `whisper-cpp` sibling crate, (C) shell out. Recommend A through Phase 6; B in Phase 7 if needed.
3. **Speculative bindings.** `common_speculative_*` lives in `llama.cpp/common`, not public `llama.h`. Need FFI shims in `llama-cpp-sys` (same `llama_rs_*` pattern as chat / JSON-schema).
4. **Jinja engine.** C++ uses `minja`. Options: bind `minja` via FFI (preferred — exact parity) or use `minijinja` (risk of template incompat). Recommend FFI shim.
5. **Continuous-batching correctness.** Highest-risk piece. Mitigation: port `update_slots()` faithfully; add deterministic unit tests for prompt-prefix reuse.
6. **Single broadcast channel per task.** Switch to per-task `mpsc` to avoid lossy delivery on slow subscribers.
7. **`AppState` cloning.** Ensure `LlamaModel` lives behind `Arc`; verify `LlamaContext` `Send` properties under chosen build features.
8. **Tower-http auth-layer.** Keep custom middleware; drop unused `tower-http` `auth` feature.
9. **Prompt-cache disk format.** Use llama.cpp session binary format directly for cross-compat with C++ slot save files (simplifies golden tests).
10. **SSE backpressure.** Bounded per-request channel + drop policy on slow client (close stream with `client_disconnected`).