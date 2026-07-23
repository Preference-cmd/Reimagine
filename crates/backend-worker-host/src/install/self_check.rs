use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use tracing;

use super::error::{InstallError, InstallResult};
use crate::ExpectedWorkerIdentity;

/// Configuration for the self-check step.
#[derive(Clone, Debug)]
pub struct SelfCheckConfig {
    /// Maximum time to wait for the worker to produce output.
    pub timeout: Duration,
    /// Maximum bytes of stderr to capture.
    pub max_stderr_bytes: usize,
}

impl Default for SelfCheckConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(5),
            max_stderr_bytes: 64 * 1024,
        }
    }
}

/// Runs a worker binary in a sterile environment to verify its identity
/// matches the signed package metadata.
///
/// The self-check:
/// - Spawns the worker with `--version` or equivalent identity flag
/// - Provides a minimal environment (no workspace, no model roots)
/// - Does NOT grant network access or catalog endpoints
/// - Captures stderr within a bounded buffer
/// - Uses a strict timeout with SIGKILL escalation
/// - Verifies identity fields match the expected values
pub struct SelfCheckRunner {
    config: SelfCheckConfig,
}

impl SelfCheckRunner {
    #[must_use]
    pub fn new(config: SelfCheckConfig) -> Self {
        Self { config }
    }

    /// Run the worker self-check.
    ///
    /// The worker is expected to produce output containing its identity
    /// metadata. This is validated against the provided `expected` identity.
    pub fn run_check(
        &self,
        executable_path: &Path,
        expected: &ExpectedWorkerIdentity,
    ) -> InstallResult<()> {
        let mut child = Command::new(executable_path)
            .arg("--version")
            .env_clear()
            .env("WORKER_SELF_CHECK", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| InstallError::SelfCheckFailed {
                message: format!("failed to spawn worker for self-check: {e}"),
            })?;
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        let stdout_reader = std::thread::spawn(move || read_bounded(stdout, 1024 * 1024));
        let stderr_limit = self.config.max_stderr_bytes;
        let stderr_reader = std::thread::spawn(move || read_bounded(stderr, stderr_limit));
        let deadline = Instant::now() + self.config.timeout;
        let status = loop {
            if let Some(status) = child
                .try_wait()
                .map_err(|e| InstallError::SelfCheckFailed {
                    message: format!("failed waiting for worker self-check: {e}"),
                })?
            {
                break status;
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_reader.join();
                let _ = stderr_reader.join();
                return Err(InstallError::SelfCheckTimeout);
            }
            std::thread::sleep(Duration::from_millis(5));
        };
        let stdout_bytes = stdout_reader.join().unwrap_or_default();
        let stderr_bytes = stderr_reader.join().unwrap_or_default();

        // Parse stdout for identity
        let stdout = String::from_utf8_lossy(&stdout_bytes);
        let stderr_preview = String::from_utf8_lossy(&stderr_bytes);

        if !status.success() {
            return Err(InstallError::SelfCheckFailed {
                message: format!(
                    "self-check exited with status {}: stderr: {}",
                    status, stderr_preview
                ),
            });
        }

        let actual =
            serde_json::from_str::<ExpectedWorkerIdentity>(stdout.trim()).map_err(|e| {
                InstallError::SelfCheckIdentityMismatch {
                    field: "identity_json".to_string(),
                    expected: format!("{expected:?}"),
                    actual: format!("invalid JSON ({e}): {stdout}"),
                }
            })?;
        if &actual != expected {
            return Err(InstallError::SelfCheckIdentityMismatch {
                field: "identity".to_string(),
                expected: format!("{expected:?}"),
                actual: format!("{actual:?}"),
            });
        }

        tracing::info!(
            "self-check passed for worker `{}` at `{}`",
            expected.installation_id.0,
            executable_path.display()
        );

        Ok(())
    }
}

fn read_bounded(mut reader: impl Read, limit: usize) -> Vec<u8> {
    let mut output = Vec::new();
    let _ = reader.by_ref().take(limit as u64).read_to_end(&mut output);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_check_runner_construction() {
        let runner = SelfCheckRunner::new(SelfCheckConfig::default());
        assert_eq!(runner.config.timeout, Duration::from_secs(5));
    }

    #[test]
    fn self_check_fails_for_missing_executable() {
        let runner = SelfCheckRunner::new(SelfCheckConfig::default());
        let expected = ExpectedWorkerIdentity {
            backend_instance_id: crate::BackendInstanceId("burn:wgpu:default".to_string()),
            installation_id: crate::WorkerInstallationId("test".to_string()),
            backend_kind: "burn".to_string(),
            target: "aarch64-apple-darwin".to_string(),
            manifest_digest: "abc".to_string(),
        };

        let result = runner.run_check(Path::new("/nonexistent/worker"), &expected);
        assert!(result.is_err());
    }

    #[cfg(unix)]
    #[test]
    fn self_check_timeout_kills_hung_process() {
        use std::os::unix::fs::PermissionsExt;
        let temp = tempfile::tempdir().unwrap();
        let executable = temp.path().join("hung-worker");
        std::fs::write(&executable, "#!/bin/sh\nsleep 5\n").unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755)).unwrap();
        let runner = SelfCheckRunner::new(SelfCheckConfig {
            timeout: Duration::from_millis(20),
            max_stderr_bytes: 1024,
        });
        let expected = ExpectedWorkerIdentity {
            backend_instance_id: crate::BackendInstanceId("burn:wgpu:default".to_owned()),
            installation_id: crate::WorkerInstallationId("test".to_owned()),
            backend_kind: "burn".to_owned(),
            target: "test-target".to_owned(),
            manifest_digest: "digest".to_owned(),
        };
        assert!(matches!(
            runner.run_check(&executable, &expected),
            Err(InstallError::SelfCheckTimeout)
        ));
    }
}
