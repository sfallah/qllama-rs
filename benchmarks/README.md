# qllama-bench

Criterion benchmarks and model-backed tests for `qllama`: embedding throughput (single, batched,
per-sentence), tokenisation (llama.cpp vs Hugging Face `tokenizers`, serial and parallel) and
cross-encoder reranking, plus `process-data`, a small CLI that embeds the texts under
`tests/test_data/` and writes the vectors as safetensors for offline evaluation.

This crate is **excluded from the root workspace**: it depends on `fast-text-splitter`, which is
still served from a private GitLab remote, and a workspace member with such a dependency would
break `cargo build --workspace` for anyone without access. Build it **from this directory** —
cargo reads `.cargo/config.toml` from the working directory, not from `--manifest-path`, and this
crate's config is what shares the root `target/` and fetches git dependencies with the system git:

```sh
cd benchmarks
cargo bench --no-run                      # compile only
cargo bench                               # needs the models below
cargo test
cargo run --bin process-data -- --help
```

Backends are the usual `qllama` features; `metal` is the default here and is a no-op off macOS, so
on a CUDA machine use `--features cuda`. Fetching `fast-text-splitter` needs an SSH key that GitLab
accepts.

Models are fetched through `hf-hub` on first use (bge-m3, bge-reranker-v2-m3, jina-reranker,
Qwen3 reranker GGUFs) and cached under `~/.cache/huggingface`; `models/` and `output/` are ignored
by git. Formerly the standalone repository `llama-cpp-rs-bench` (GitLab); its history is merged
into this repository under this directory.
