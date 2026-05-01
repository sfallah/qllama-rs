# llama-server-rs Endpoint Specification (Phase 1)

This document is the parity target for `llama.cpp/tools/server` REST APIs (frontend excluded).

## Legend
- Auth: `public` means API key not required in llama.cpp, `api-key` means protected when keys configured.
- Slot/Queue: expected interaction with scheduler/slots in Phase 2.
- SSE events: expected server-sent event stream names/chunks for streaming endpoints.

## Endpoint Matrix

| Method | Path | Request schema (summary) | Response schema (summary) | SSE event names | Auth | Slot/Queue interaction | C++ source refs |
|---|---|---|---|---|---|---|---|
| GET | `/health` | none | `{status}` or health object | none | public | none | `tools/server/server.cpp:172`, `tools/server/server-http.cpp:36-41` |
| GET | `/v1/health` | none | same as `/health` | none | public | none | `tools/server/server.cpp:173` |
| GET | `/metrics` | none | Prometheus text exposition | none | api-key | read-only queue/slot counters | `tools/server/server.cpp:174` |
| GET | `/props` | none | model + defaults + runtime props | none | api-key | reads queue + slot defaults | `tools/server/server.cpp:175` |
| POST | `/props` | partial runtime config patch | updated props | none | api-key | updates server defaults, no task enqueue | `tools/server/server.cpp:176` |
| GET | `/models` | none | OpenAI-style model list | none | public | reads model registry/router registry | `tools/server/server.cpp:177` |
| GET | `/v1/models` | none | same as `/models` | none | public | reads model registry/router registry | `tools/server/server.cpp:178` |
| POST | `/completion` | legacy completion payload (`prompt`, sampling params, `stream?`) | legacy completion object | token chunk + stop object (legacy JSON stream) | api-key | enqueues generation task on slot queue | `tools/server/server.cpp:179`, `tools/server/server-common.cpp:813+` |
| POST | `/completions` | same as `/completion` | same as `/completion` | same as `/completion` | api-key | enqueues generation task on slot queue | `tools/server/server.cpp:180` |
| POST | `/v1/completions` | OpenAI completions payload | OpenAI completions response | `data: {id,choices[*].text...}` + `[DONE]` | api-key | enqueues generation task on slot queue | `tools/server/server.cpp:181` |
| POST | `/chat/completions` | OpenAI chat payload | chat completion response | chat delta chunks + final usage + `[DONE]` | api-key | enqueues chat task; slot assignment + cache reuse | `tools/server/server.cpp:182`, `tools/server/server-common.cpp:910+` |
| POST | `/v1/chat/completions` | same as `/chat/completions` | same as `/chat/completions` | same as `/chat/completions` | api-key | enqueues chat task; slot assignment + cache reuse | `tools/server/server.cpp:183` |
| POST | `/responses` | OpenAI Responses API payload | OpenAI Responses API response | Responses deltas (compat mode) + `[DONE]` | api-key | enqueues chat-compatible task | `tools/server/server.cpp:185` |
| POST | `/v1/responses` | same as `/responses` | same as `/responses` | same as `/responses` | api-key | enqueues chat-compatible task | `tools/server/server.cpp:184` |
| POST | `/audio/transcriptions` | multipart/audio transcription payload | OpenAI transcription response | none | api-key | enqueues transcription task | `tools/server/server.cpp:187` |
| POST | `/v1/audio/transcriptions` | same as `/audio/transcriptions` | same as `/audio/transcriptions` | none | api-key | enqueues transcription task | `tools/server/server.cpp:186` |
| POST | `/v1/messages` | Anthropic Messages payload | Anthropic response object | anthropic event stream (`message_start`, `content_block_*`, `message_delta`, etc.) | api-key | enqueues chat-compatible task | `tools/server/server.cpp:188` |
| POST | `/v1/messages/count_tokens` | Anthropic token counting payload | `{input_tokens}` | none | api-key | no generation; prompt build/tokenize only | `tools/server/server.cpp:189` |
| POST | `/infill` | FIM payload (`input_prefix/suffix`, params, `stream?`) | infill completion object | token chunks + stop | api-key | enqueues infill task | `tools/server/server.cpp:190` |
| POST | `/embedding` | legacy embedding payload | embedding vectors | none | api-key | enqueue embedding batch task (or immediate if idle) | `tools/server/server.cpp:191` |
| POST | `/embeddings` | embedding payload (`input`) | embeddings response | none | api-key | enqueue embedding batch task | `tools/server/server.cpp:192` |
| POST | `/v1/embeddings` | OpenAI embeddings payload | OpenAI embeddings response (`float`/`base64`) | none | api-key | enqueue embedding batch task | `tools/server/server.cpp:193` |
| POST | `/rerank` | rerank payload (`query`, `documents` or TEI `texts`) | rerank result list | none | api-key | enqueue rerank scoring task | `tools/server/server.cpp:194` |
| POST | `/reranking` | alias of `/rerank` | same | none | api-key | enqueue rerank scoring task | `tools/server/server.cpp:195` |
| POST | `/v1/rerank` | alias of `/rerank` | same | none | api-key | enqueue rerank scoring task | `tools/server/server.cpp:196` |
| POST | `/v1/reranking` | alias of `/rerank` | same | none | api-key | enqueue rerank scoring task | `tools/server/server.cpp:197` |
| POST | `/tokenize` | `{content, add_special?, with_pieces?}` | `{tokens:[...]}` | none | api-key | no queue; direct tokenizer op | `tools/server/server.cpp:198` |
| POST | `/detokenize` | `{tokens:[int...]}` | `{content}` | none | api-key | no queue; direct tokenizer op | `tools/server/server.cpp:199` |
| POST | `/apply-template` | `{messages,...template opts}` | `{prompt,...meta}` | none | api-key | no queue; template render only | `tools/server/server.cpp:200` |
| GET | `/lora-adapters` | none | loaded adapter list | none | api-key | reads active adapters in model context | `tools/server/server.cpp:202` |
| POST | `/lora-adapters` | adapter set/update payload | adapter state | none | api-key | mutates adapter stack; may gate active tasks | `tools/server/server.cpp:203` |
| GET | `/slots` | none | slot states array | none | api-key | reads scheduler/slot runtime | `tools/server/server.cpp:205` |
| POST | `/slots/:id_slot` | action query (`save/restore/erase`) + optional body | action result | none | api-key | mutates slot persistent state | `tools/server/server.cpp:206` |
| POST | `/models/load` | router model load payload | load ack | none | api-key | router operation; no local slot task | `tools/server/server.cpp:168` |
| POST | `/models/unload` | router model unload payload | unload ack | none | api-key | router operation; may terminate child slots | `tools/server/server.cpp:169` |
| GET | `/cors-proxy` | proxy query | proxied response | none | api-key | no slots (proxy utility) | `tools/server/server.cpp:211` |
| POST | `/cors-proxy` | proxy body | proxied response | none | api-key | no slots (proxy utility) | `tools/server/server.cpp:212` |
| GET | `/tools` | none | built-in tools metadata | none | api-key | no slots (tool registry) | `tools/server/server.cpp:223` |
| POST | `/tools` | tool invocation payload | tool output | none | api-key | no model slot unless tool triggers model op | `tools/server/server.cpp:224` |

## Shared Request/Response Schema Notes

- OpenAI completions/chat/responses fields track llama.cpp compatibility options:
  - `temperature`, `top_p`, `top_k`, `seed`, `max_tokens`/`n_predict`, `stop`, `stream`.
- Chat-specific compatibility:
  - `tools`, `tool_choice`, `response_format`, `json_schema`, `grammar`, `chat_template`, `chat_template_kwargs`, `reasoning_format`, `parallel_tool_calls`, `enable_thinking`.
- Embeddings:
  - `input: string | array<string|token-array>`, `encoding_format: float|base64` (OpenAI flavor).
- Rerank:
  - native format: `{query, documents, top_n?}`
  - TEI format: `{query, texts, top_n?}`.

## Streaming Behavior Contract (Phase 2 Target)

- `/completion` legacy stream: NDJSON chunks with token text + final stop/timings chunk.
- `/v1/completions`: OpenAI SSE with `data: {choices:[{text,...}]}` and terminal `data: [DONE]`.
- `/chat/completions` + `/v1/chat/completions`: SSE deltas (`choices[*].delta`) + optional usage event + `[DONE]`.
- `/responses` + `/v1/responses`: OpenAI Responses SSE delta events and terminal completion event.
- `/v1/messages`: Anthropic event stream (`message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`).

## Security / Auth Contract

- API key validation follows llama.cpp behavior from `server-http.cpp`:
  - Public endpoints: `/health`, `/v1/health`, `/models`, `/v1/models` (+ web static assets in C++, not used here).
  - All other endpoints require valid `Authorization: Bearer <key>` or `X-Api-Key` when keys configured.
