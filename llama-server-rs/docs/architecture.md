# llama-server-rs Architecture (Phase 2 Design)

This document defines the target production architecture for full llama.cpp server API parity in Rust.

## Goals
- Full REST + SSE parity with `llama.cpp/tools/server` (frontend excluded).
- Deterministic slot scheduling and queue behavior compatible with llama.cpp semantics.
- OpenAI + Anthropic compatibility endpoints, including streaming and tool-call parsing.
- Clear separation between transport (Axum), orchestration (queue/scheduler), and inference execution.

## High-Level Components

1. HTTP Layer (Axum)
- `Router` registers all API routes and aliases.
- Middleware stack:
  - CORS (`tower-http::cors`)
  - tracing (`tower-http::trace`)
  - auth validation (`Authorization` / `X-Api-Key`)
  - request ID propagation
- SSE responses implemented with `axum::response::sse::Sse` + `async-stream`.

2. API Compatibility Layer
- Endpoint-specific adapters map input/output to internal task types:
  - OpenAI completions/chat/responses
  - Anthropic messages/count_tokens
  - legacy completion/embedding endpoints
- Compatibility mappers normalize aliases and schema variations.

3. Task Queue + Scheduler
- Global MPSC queue of `ServerTask` items.
- Slot scheduler assigns tasks to inference slots based on:
  - endpoint type (gen/embed/rerank/tokenize)
  - slot availability
  - prompt cache reuse affinity
  - explicit slot id where applicable (`/slots/:id_slot` flows)
- Priority policy:
  - control tasks (health/props/models/lora/slots ops) bypass or use high-priority lane
  - generation tasks go through fair queue with cancellation support

4. Slot Runtime
- `SlotState` per slot (DashMap/ParkingLot):
  - active task id
  - KV cache metadata
  - prompt cache stats
  - save/restore snapshots metadata
  - last-used timestamp for idle policies
- `SlotWorker` task per slot drives llama decode loop and emits stream deltas.

5. Model Runtime
- `ModelRegistry` for single-model mode + router mode abstraction.
- `ModelHandle` owns loaded model/context factories and adapter stack.
- Router mode (Phase 2b): model process map + proxy dispatch parity with llama.cpp router endpoints.

6. Streaming/Event Pipeline
- Internal event enum `TaskEvent` converted to endpoint-specific SSE shapes.
- Backpressure-aware channel per request.
- Unified stop/finalization logic emits usage and final stop reason consistently.

7. Observability
- Prometheus registry:
  - request counters/latency by endpoint
  - queue depth
  - active slots
  - token throughput
  - error counts by type/status
- `/metrics` exports text format.

8. TLS/Networking
- Plain HTTP via `axum::serve`.
- Optional TLS via `axum-server` + rustls cert/key.

## Core Data Types

- `AppState`
  - shared config, auth config, model registry, scheduler, metrics registry.
- `ServerTask`
  - task id (UUID), kind, normalized params, stream flag, cancellation token.
- `TaskKind`
  - `Completion`, `ChatCompletion`, `Responses`, `Embedding`, `Rerank`, `Tokenize`, `Detokenize`, `ApplyTemplate`, `Infill`, `Transcription`, etc.
- `TaskResult`
  - terminal result or error metadata.
- `TaskEvent`
  - token delta, tool delta, usage delta, final stop event.

## Request Lifecycle

1. Axum handler validates auth and schema.
2. Compatibility mapper converts payload into canonical params.
3. Handler enqueues task and either:
  - awaits terminal result for non-streaming endpoints, or
  - returns SSE stream bound to task event channel.
4. Scheduler assigns task to slot worker.
5. Worker executes qllama calls and emits events.
6. Handler adapter maps events into endpoint-specific response frames.

## Error Model
- Strong typed errors via `thiserror`:
  - `InvalidRequest`, `AuthError`, `ModelError`, `SchedulerError`, `StreamingError`, `InternalError`.
- Error-to-HTTP mapping keeps compatibility fields:
  - `{"error": {"message", "type", "code"}}`
- Streaming errors emit final error event + close stream.

## Concurrency Model
- Tokio multi-thread runtime.
- Slot workers run as long-lived tasks.
- Queue coordination uses async channels + lock-minimized state maps.
- Blocking or CPU-heavy operations run via `spawn_blocking` when needed.

## Phase 2 Implementation Sequence

1. Transport parity
- Replace scaffold handlers with full typed request/response handlers.
- Add SSE pipelines for all streaming endpoints.

2. Scheduler + slots
- Implement queue, slot assignment, and task cancellation.
- Implement `/slots` + `/slots/:id_slot` save/restore/erase.

3. Endpoint completion
- completions/chat/responses parity
- embeddings/rerank/tokenize/detokenize/apply-template parity
- anthropic compatibility parity
- lora adapter management parity

4. Router mode
- Implement `/models/load` and `/models/unload` orchestration and proxying.

5. Hardening
- metrics completeness, soak tests, compatibility fixtures, TLS smoke tests.
