# qllama-sys

Raw bindgen bindings to llama.cpp, plus the CMake build of the vendored llama.cpp submodule.
Backends are selected with the `cuda`, `metal`, `vulkan`, `openmp` and `rocm` features; `mtmd` adds
the multimodal library.

See [qllama](https://crates.io/crates/qllama) for the safe API.
