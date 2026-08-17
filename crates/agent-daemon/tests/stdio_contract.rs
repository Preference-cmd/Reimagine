//! Binary-level stdio contract for the frozen daemon (AR-12).
//!
//! The daemon speaks JSON-RPC 2.0 over stdio. Log lines written to
//! stdout would corrupt that stream, so tracing must target stderr and
//! stdout must contain only newline-delimited JSON-RPC.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn daemon_bin() -> &'static str {
    env!("CARGO_BIN_EXE_reimagine-agent-daemon")
}

fn temp_dir() -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-agent-daemon-stdio-{nonce}"))
}

#[test]
fn binary_tracing_goes_to_stderr_and_stdout_stays_jsonrpc_only() {
    let workspace_dir = temp_dir();
    let mut child = Command::new(daemon_bin())
        .arg("--workspace-dir")
        .arg(&workspace_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("daemon binary spawns");

    let mut stdout = child.stdout.take().expect("stdout piped");
    let mut stderr = child.stderr.take().expect("stderr piped");

    // stdin is closed immediately, so the daemon should serve the empty
    // stream and exit. Wait generously: workspace bootstrap is the slow
    // part and runs on cold caches in CI.
    let deadline = Instant::now() + Duration::from_secs(120);
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll daemon process") {
            break status;
        }
        assert!(Instant::now() < deadline, "daemon did not exit within 120s");
        std::thread::sleep(Duration::from_millis(200));
    };
    assert!(status.success(), "daemon exits cleanly: {status}");

    let mut stdout_text = String::new();
    stdout
        .read_to_string(&mut stdout_text)
        .expect("read daemon stdout");

    // Whatever the daemon writes to stdout must be JSON-RPC lines.
    for line in stdout_text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|error| panic!("stdout line is not JSON: {line:?} ({error})"));
        assert_eq!(
            value.get("jsonrpc").and_then(serde_json::Value::as_str),
            Some("2.0"),
            "stdout line is not a JSON-RPC envelope: {line:?}"
        );
    }

    let mut stderr_text = String::new();
    stderr
        .read_to_string(&mut stderr_text)
        .expect("read daemon stderr");
    assert!(
        stderr_text.contains("agent daemon started"),
        "tracing should reach stderr, got: {stderr_text}"
    );

    let _ = std::fs::remove_dir_all(&workspace_dir);
}
