# qllama-rs

Rust bindings to [llama.cpp](https://github.com/ggml-org/llama.cpp), built for production
embedding, reranking and completion services.

Two crates:

- **`qllama`** — the safe API. This is what applications depend on.
- **`qllama-sys`** — bindgen bindings plus the build of the vendored llama.cpp. llama.cpp is a git
  submodule at `qllama-sys/llama.cpp`, compiled from source with CMake at build time.

`qllama-rs` started as a fork of [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)
and keeps its MIT/Apache-2.0 licensing and history, but it has diverged: the vendored llama.cpp is
pinned to stable releases, the API is being reshaped around the needs of the services that use it,
and the crate names are its own. It is not a drop-in replacement for `llama-cpp-2`.

## Building

`qllama-sys` compiles llama.cpp from source, so you need `cmake` and a C/C++ toolchain. The first
build is slow. Backends are Cargo features on `qllama`: `cuda`, `metal`, `vulkan`, `openmp` (default).
On Apple Silicon, Metal is enabled regardless of the feature flag.

Clone with submodules:

```sh
git clone --recursive https://github.com/sfallah/qllama-rs
```

or, in an existing checkout:

```sh
git submodule update --init --recursive
```

Try the examples (add `--features cuda` on a CUDA machine):

```sh
cargo run --release -p simple -- --prompt "The way to kill a linux process is" hf-model TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf
cargo run --release -p embeddings -- --help
cargo run --release -p reranker -- --help
```

## Versioning and llama.cpp releases

Each vendored llama.cpp release lives on its own branch, named `llama.cpp_b<build>_<date>` after the
llama.cpp build tag and its date (for example `llama.cpp_b10964_2026-09-14` is llama.cpp v0.4.1).
The crates follow 0.x semver independently of the llama.cpp build number.

## License

MIT or Apache-2.0, at your option — see `LICENSE-MIT` and `LICENSE-APACHE`. llama.cpp itself is MIT
licensed and is vendored unmodified.
