//! Agent daemon binary entry point (frozen experimental sidecar).
//!
//! Parses `--workspace-dir`, boots the daemon workspace, serves the
//! JSON-RPC loop over stdin/stdout, and runs graceful shutdown when
//! stdin reaches EOF: pending notifications are already flushed by the
//! transport, every live session context is persisted, and the
//! workspace's inference workers are shut down.
//!
//! Stdio contract: **stdout carries only newline-delimited JSON-RPC**.
//! All tracing/diagnostics go to stderr so protocol parsers never see
//! log lines mixed into the JSON stream.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use tracing_subscriber::EnvFilter;

use reimagine_agent_daemon::server::AgentDaemon;

/// Command-line arguments for the daemon binary.
#[derive(Debug, Parser)]
#[command(
    name = "reimagine-agent-daemon",
    version,
    about = "Reimagine agent daemon: JSON-RPC 2.0 over stdio"
)]
struct Cli {
    /// Directory the daemon workspace is rooted at (created on boot).
    ///
    /// Defaults to `./workspace` relative to the current directory.
    #[arg(long, default_value = "./workspace")]
    workspace_dir: PathBuf,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    init_tracing();
    daemon_main(cli)
}

#[tokio::main]
async fn daemon_main(cli: Cli) -> ExitCode {
    let mut daemon = match AgentDaemon::new(&cli.workspace_dir).await {
        Ok(daemon) => daemon,
        Err(error) => {
            tracing::error!(
                path = %cli.workspace_dir.display(),
                %error,
                "daemon initialization failed"
            );
            return ExitCode::FAILURE;
        }
    };
    tracing::info!(
        path = %cli.workspace_dir.display(),
        "agent daemon started; serving JSON-RPC over stdio"
    );

    let serve_result = daemon.serve_stdio().await;
    if let Err(error) = &serve_result {
        tracing::error!(%error, "stdio serve loop failed");
    }

    shutdown(&daemon).await;
    tracing::info!("agent daemon exited cleanly");

    match serve_result {
        Ok(()) => ExitCode::SUCCESS,
        Err(_) => ExitCode::FAILURE,
    }
}

/// Graceful shutdown after stdin EOF.
///
/// Pending notifications need no extra flush: the transport writes every
/// JSON line under a single write lock and flushes after each line, so
/// nothing can be left buffered at EOF. Sessions are persisted next, and
/// finally the workspace shuts its inference workers down so no child
/// processes are orphaned.
async fn shutdown(daemon: &AgentDaemon) {
    let persisted = daemon.persist_all().await;
    tracing::info!(persisted, "persisted daemon sessions");
    daemon.workspace().host().shutdown().await;
}

/// Initialize the process-wide tracing subscriber.
///
/// Filtering comes from `RUST_LOG` when set; otherwise the default
/// `info` level applies. Production output goes to **stderr** so the
/// JSON-RPC stream on stdout stays clean.
fn init_tracing() {
    init_tracing_with_writer(std::io::stderr);
}

/// Initialize the tracing subscriber writing to `writer`.
///
/// Split out from [`init_tracing`] so tests can point the subscriber at
/// a capture buffer and prove logs never target the default stdout
/// writer.
fn init_tracing_with_writer<W>(writer: W)
where
    W: for<'a> tracing_subscriber::fmt::MakeWriter<'a> + Send + Sync + 'static,
{
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(writer)
        .init();
}

#[cfg(test)]
mod tests {
    use std::io::Write as IoWrite;
    use std::sync::{Arc, Mutex};

    use super::*;

    /// `MakeWriter` that captures formatted output into a shared byte
    /// buffer. Used to prove the tracing subscriber targets the writer
    /// handed to it instead of defaulting to stdout.
    #[derive(Clone, Default)]
    struct CaptureWriter(Arc<Mutex<Vec<u8>>>);

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CaptureWriter {
        type Writer = CaptureGuard<'a>;

        fn make_writer(&'a self) -> Self::Writer {
            CaptureGuard {
                guard: self
                    .0
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            }
        }
    }

    struct CaptureGuard<'a> {
        guard: std::sync::MutexGuard<'a, Vec<u8>>,
    }

    impl IoWrite for CaptureGuard<'_> {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.guard.write(buf)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.guard.flush()
        }
    }

    #[test]
    fn tracing_writes_to_configured_writer_instead_of_default_stdout() {
        let capture = CaptureWriter::default();
        init_tracing_with_writer(capture.clone());
        tracing::error!("stdio-contract-marker");

        let bytes = capture
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let output = String::from_utf8(bytes).expect("captured output is utf8");
        assert!(
            output.contains("stdio-contract-marker"),
            "tracing output should reach the configured writer, got: {output}"
        );
    }

    #[test]
    fn cli_parses_workspace_dir() {
        let cli = Cli::try_parse_from([
            "reimagine-agent-daemon",
            "--workspace-dir",
            "/tmp/reimagine-ws",
        ])
        .expect("cli parses");
        assert_eq!(cli.workspace_dir, PathBuf::from("/tmp/reimagine-ws"));
    }

    #[test]
    fn cli_defaults_to_dot_workspace() {
        let cli = Cli::try_parse_from(["reimagine-agent-daemon"]).expect("cli parses");
        assert_eq!(cli.workspace_dir, PathBuf::from("./workspace"));
    }

    #[test]
    fn cli_rejects_unknown_flags() {
        assert!(Cli::try_parse_from(["reimagine-agent-daemon", "--bogus"]).is_err());
    }
}
