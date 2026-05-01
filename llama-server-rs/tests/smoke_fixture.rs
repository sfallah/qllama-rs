use std::path::Path;

#[test]
fn tiny_gguf_fixture_is_readable() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../llama-cpp/src/gguf/ggml-vocab-bert-bge.gguf");

    assert!(
        fixture.exists(),
        "fixture not found at {}",
        fixture.display()
    );

    let ctx = llama_cpp::gguf::GgufContext::from_file(&fixture);
    assert!(
        ctx.is_some(),
        "failed to load gguf fixture at {}",
        fixture.display()
    );
}
