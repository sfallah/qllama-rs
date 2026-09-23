//! Shared harness for the HTTP integration tests.
//!
//! Every test spawns its own `llama-server-rs` process on its own free port so
//! that the suite can run on parallel threads against a serial engine.
#![allow(dead_code, missing_docs)]

use std::io::{Read, Seek, SeekFrom};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use hf_hub::api::sync::ApiBuilder;
use serde_json::Value;

/// Repo of the ~1.1 MB generation-test model (resolved the same way
/// `config::resolve_model` resolves `hf-model`).
pub const MODEL_REPO: &str = "ggml-org/models";
/// File inside [`MODEL_REPO`] holding the tiny stories model.
pub const MODEL_FILE: &str = "tinyllamas/stories260K.gguf";
/// Context size used with the tiny model; anything larger just wastes time.
pub const TINY_CTX: &str = "512";

/// Backend + model init can take several seconds on a cold GPU, so be generous.
const READY_TIMEOUT: Duration = Duration::from_mins(2);
/// A server whose model never loads still answers `/health` almost immediately.
const RESPONDING_TIMEOUT: Duration = Duration::from_mins(1);
const POLL_INTERVAL: Duration = Duration::from_millis(50);
/// How much of the captured server log to include in a failure message.
const LOG_TAIL_BYTES: usize = 8192;

/// Binds port 0 and immediately releases it, so the child gets a port that is
/// (almost certainly) still free when it binds a moment later.
pub fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind an ephemeral port");
    let port = listener
        .local_addr()
        .expect("read the ephemeral port")
        .port();
    drop(listener);
    port
}

/// A running `llama-server-rs` child process plus a client pointed at it.
#[derive(Debug)]
pub struct TestServer {
    child: Child,
    /// stdout+stderr of the child, kept on disk so a chatty backend cannot fill
    /// a pipe buffer and deadlock the server.
    log: std::fs::File,
    base_url: String,
    pub client: reqwest::Client,
}

impl TestServer {
    /// Spawns a server for `model` and waits for `/health` to report 200.
    pub async fn with_model(model: &Path, extra_args: &[&str]) -> Self {
        Self::spawn(model, extra_args, true).await
    }

    /// Spawns a server whose model can never load and waits only for the HTTP
    /// listener to answer.
    pub async fn without_model(extra_args: &[&str]) -> Self {
        Self::spawn(Path::new("/nonexistent-model.gguf"), extra_args, false).await
    }

    async fn spawn(model: &Path, extra_args: &[&str], wait_ready: bool) -> Self {
        let port = free_port();
        let log = tempfile::tempfile().expect("create the server log file");
        let child = Command::new(env!("CARGO_BIN_EXE_llama-server-rs"))
            .args(["--host", "127.0.0.1", "--port", &port.to_string()])
            .args(extra_args)
            .arg("local")
            .arg(model)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log.try_clone().expect("clone the log handle")))
            .stderr(Stdio::from(log.try_clone().expect("clone the log handle")))
            .spawn()
            .expect("spawn llama-server-rs");

        let mut server = Self {
            child,
            log,
            base_url: format!("http://127.0.0.1:{port}"),
            client: reqwest::Client::builder()
                .build()
                .expect("build the test client"),
        };
        server.wait_until(wait_ready).await;
        server
    }

    async fn wait_until(&mut self, wait_ready: bool) {
        let timeout = if wait_ready {
            READY_TIMEOUT
        } else {
            RESPONDING_TIMEOUT
        };
        let deadline = Instant::now() + timeout;
        loop {
            if let Ok(Some(status)) = self.child.try_wait() {
                panic!("server exited early with {status}\n{}", self.log_tail());
            }
            let response = self
                .client
                .get(self.url("/health"))
                .timeout(Duration::from_secs(5))
                .send()
                .await;
            if let Ok(response) = response {
                if !wait_ready || response.status() == reqwest::StatusCode::OK {
                    return;
                }
            }
            assert!(
                Instant::now() < deadline,
                "server was not {} within {timeout:?}\n{}",
                if wait_ready { "ready" } else { "reachable" },
                self.log_tail()
            );
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    pub fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base_url)
    }

    /// POSTs `body` as JSON without asserting anything about the outcome.
    pub async fn post(&self, path: &str, body: &Value) -> reqwest::Response {
        self.client
            .post(self.url(path))
            .json(body)
            .send()
            .await
            .unwrap_or_else(|err| panic!("POST {path} failed: {err}"))
    }

    /// POSTs `body` as JSON and decodes a successful response.
    pub async fn post_json(&self, path: &str, body: &Value) -> Value {
        let response = self.post(path, body).await;
        let status = response.status();
        let text = response.text().await.expect("read the response body");
        assert!(status.is_success(), "POST {path} -> {status}: {text}");
        serde_json::from_str(&text)
            .unwrap_or_else(|err| panic!("POST {path} returned non-JSON ({err}): {text}"))
    }

    pub async fn get_text(&self, path: &str) -> String {
        self.client
            .get(self.url(path))
            .send()
            .await
            .unwrap_or_else(|err| panic!("GET {path} failed: {err}"))
            .text()
            .await
            .expect("read the response body")
    }

    /// Value of a plaintext `/metrics` counter.
    pub async fn metric(&self, name: &str) -> u64 {
        parse_metric(&self.get_text("/metrics").await, name)
    }

    pub fn log_tail(&self) -> String {
        let Ok(mut file) = self.log.try_clone() else {
            return "<server log unavailable>".to_string();
        };
        if file.seek(SeekFrom::Start(0)).is_err() {
            return "<server log unavailable>".to_string();
        }
        let mut raw = Vec::new();
        if file.read_to_end(&mut raw).is_err() {
            return "<server log unavailable>".to_string();
        }
        let start = raw.len().saturating_sub(LOG_TAIL_BYTES);
        format!(
            "--- server log tail ---\n{}",
            String::from_utf8_lossy(&raw[start..])
        )
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Reads a counter out of the plaintext `/metrics` body.
pub fn parse_metric(body: &str, name: &str) -> u64 {
    for line in body.lines() {
        if let Some(value) = line.strip_prefix(name).and_then(|r| r.strip_prefix(' ')) {
            return value
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("metric {name} is not an integer: {line}"));
        }
    }
    panic!("metric {name} is missing from:\n{body}")
}

/// Splits a fully buffered SSE body into the payload of each `data:` frame.
pub fn sse_data_frames(body: &str) -> Vec<String> {
    body.split("\n\n")
        .filter(|frame| !frame.is_empty())
        .map(|frame| {
            frame
                .strip_prefix("data: ")
                .unwrap_or_else(|| panic!("not an SSE data frame: {frame:?}"))
                .to_string()
        })
        .collect()
}

/// Same as [`sse_data_frames`] but decodes every frame as JSON.
pub fn sse_json_frames(body: &str) -> Vec<Value> {
    sse_data_frames(body)
        .iter()
        .map(|frame| {
            serde_json::from_str(frame)
                .unwrap_or_else(|err| panic!("SSE frame is not JSON ({err}): {frame}"))
        })
        .collect()
}

/// Path to a small GGUF suitable for generation tests, or `None` (with a reason
/// on stderr) when neither the env override nor the HF cache can supply one.
pub fn test_model() -> Option<PathBuf> {
    static MODEL: OnceLock<Option<PathBuf>> = OnceLock::new();
    MODEL.get_or_init(resolve_test_model).clone()
}

fn resolve_test_model() -> Option<PathBuf> {
    if let Some(raw) = std::env::var_os("LLAMA_SERVER_TEST_MODEL") {
        let path = PathBuf::from(raw);
        if path.is_file() {
            return Some(path);
        }
        eprintln!(
            "LLAMA_SERVER_TEST_MODEL={} is not a readable file",
            path.display()
        );
        return None;
    }

    let downloaded = ApiBuilder::new()
        .with_progress(false)
        .build()
        .map_err(|err| err.to_string())
        .and_then(|api| {
            api.model(MODEL_REPO.to_string())
                .get(MODEL_FILE)
                .map_err(|err| err.to_string())
        });
    match downloaded {
        Ok(path) => Some(path),
        Err(err) => {
            eprintln!(
                "could not fetch {MODEL_REPO}/{MODEL_FILE} ({err}); \
                 set LLAMA_SERVER_TEST_MODEL to run the generation tests"
            );
            None
        }
    }
}
