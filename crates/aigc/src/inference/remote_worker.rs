//! A remote worker reached over gRPC, implementing [`SwitchableWorker`].
//!
//! A remote worker serves the `reimagine.worker.v2` contract over TLS on
//! an `https://` endpoint (self-signed certificate, TOFU certificate
//! pinning against a [`TrustedKeyStore`], bearer-token authentication —
//! the same client and contract local workers use). Remote workers are
//! registration-only today: they are registered and displayed in the
//! compute topology but never routed to.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use reimagine_backend_worker_host::{WorkerProcessState, WorkerRunLeases};
use reimagine_backend_worker_protocol::remote::connect_with_trust;
use reimagine_backend_worker_protocol::trust::TrustedKeyStore;
use reimagine_backend_worker_protocol::{GrpcAuth, WorkerClient, WorkerHello, WorkerIncarnationId};
use reimagine_inference::{
    Backend, BackendInstance, BackendInstanceSnapshot, DeviceProfile, InferenceBackend,
};

use super::switch::{SwitchableWorker, WorkerSwitchError, WorkerSwitchTarget};

/// A remote worker connected via gRPC, implementing [`SwitchableWorker`].
///
/// This wraps a connected remote worker and provides the same interface
/// as [`ProcessSwitchableWorker`] for use with [`WorkerSwitchService`].
pub struct RemoteSwitchableWorker {
    instance: BackendInstance,
    incarnation_id: WorkerIncarnationId,
    run_leases: Arc<WorkerRunLeases>,
    device_label: String,
    hello: WorkerHello,
    client: Arc<WorkerClient>,
}

impl RemoteSwitchableWorker {
    /// Create a new `RemoteSwitchableWorker` from a connected client and
    /// its hello.
    ///
    /// The device label is taken from the worker's advertised instance
    /// profile, falling back to `"remote"`.
    pub fn new(hello: WorkerHello, client: Arc<WorkerClient>) -> Self {
        let device_label = hello
            .profile
            .instances
            .first()
            .map(|instance| instance.device_label.clone())
            .unwrap_or_else(|| "remote".to_owned());
        Self {
            instance: BackendInstance::new(hello.identity.backend_instance_id.0.clone()),
            incarnation_id: hello.identity.incarnation_id.clone(),
            run_leases: Arc::new(WorkerRunLeases::new()),
            device_label,
            hello,
            client,
        }
    }

    /// Access the underlying gRPC client.
    pub fn client(&self) -> &Arc<WorkerClient> {
        &self.client
    }
}

#[async_trait]
impl SwitchableWorker for RemoteSwitchableWorker {
    fn instance(&self) -> &BackendInstance {
        &self.instance
    }

    fn incarnation_id(&self) -> &WorkerIncarnationId {
        &self.incarnation_id
    }

    fn run_leases(&self) -> &Arc<WorkerRunLeases> {
        &self.run_leases
    }

    fn process_state(&self) -> WorkerProcessState {
        // Remote workers are ready once connected and greeted.
        WorkerProcessState::Ready
    }

    fn inference_backend(&self) -> Option<Arc<dyn InferenceBackend>> {
        None // remote workers do not expose a local inference backend
    }

    async fn snapshot(&self) -> BackendInstanceSnapshot {
        BackendInstanceSnapshot {
            backend_instance: self.instance.clone(),
            backend: Backend::new(self.hello.identity.backend_kind.clone()),
            plugin: None,
            extension: None,
            // Remote workers always report DeviceKind::Remote
            device: Some(DeviceProfile::new(format!("remote:{}", self.device_label))),
            observations: std::collections::BTreeMap::from([
                ("transport".to_owned(), "grpc".to_owned()),
                ("remote_addr".to_owned(), self.client.endpoint().to_owned()),
            ]),
            diagnostics: Vec::new(),
        }
    }

    async fn shutdown(&self) -> Result<(), WorkerSwitchError> {
        self.client
            .shutdown()
            .await
            .map_err(|error| WorkerSwitchError::Shutdown {
                instance: self.instance.clone(),
                message: error.to_string(),
            })?;
        Ok(())
    }
}

/// Configuration for connecting to a remote gRPC worker.
#[derive(Clone)]
pub struct RemoteWorkerCandidateConfig {
    /// The stable worker identity the trust store pins certificates
    /// under (the mDNS instance id, or the configured endpoint id).
    pub worker_id: String,
    /// The endpoint URI, e.g. `https://192.168.1.50:9100` (remote, TLS
    /// with TOFU pinning) or `http://127.0.0.1:50051` (loopback dev).
    pub endpoint: String,
    /// Bearer-token settings. A token without TLS is refused by the
    /// client; non-loopback endpoints without TLS are refused by the
    /// candidate. An explicit `tls` mode overrides TOFU pinning for
    /// `https://` endpoints.
    pub auth: GrpcAuth,
    /// The TOFU trust store: the first connect pins the presented
    /// certificate, later connects require a match.
    pub trust_store: Arc<Mutex<TrustedKeyStore>>,
    /// Per-attempt connect timeout. Generous for cloud reachability;
    /// this bounds a single network connect + hello attempt.
    pub connect_timeout: std::time::Duration,
    /// Additional connect attempts after the first failure (0 = single
    /// attempt). Retries re-run the full connect + hello.
    pub connect_retries: u32,
}

impl std::fmt::Debug for RemoteWorkerCandidateConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteWorkerCandidateConfig")
            .field("worker_id", &self.worker_id)
            .field("endpoint", &self.endpoint)
            .field("auth", &"<GrpcAuth (redacted)>")
            .field("trust_store", &"<TrustedKeyStore>")
            .field("connect_timeout", &self.connect_timeout)
            .field("connect_retries", &self.connect_retries)
            .finish()
    }
}

/// A candidate that connects to a remote worker via gRPC.
///
/// Implements [`WorkerSwitchTarget`] to enable switching to a remote
/// worker. Remote workers are registration-only today; routing stays
/// out of scope.
pub struct RemoteWorkerCandidate {
    config: RemoteWorkerCandidateConfig,
}

impl RemoteWorkerCandidate {
    /// Create a new candidate from a configuration.
    pub fn new(config: RemoteWorkerCandidateConfig) -> Self {
        Self { config }
    }

    /// Create a new candidate with default timeout/retry settings.
    pub fn with_defaults(
        worker_id: impl Into<String>,
        endpoint: impl Into<String>,
        auth: GrpcAuth,
        trust_store: Arc<Mutex<TrustedKeyStore>>,
    ) -> Self {
        Self {
            config: RemoteWorkerCandidateConfig {
                worker_id: worker_id.into(),
                endpoint: endpoint.into(),
                auth,
                trust_store,
                connect_timeout: std::time::Duration::from_secs(10),
                connect_retries: 2,
            },
        }
    }

    /// Connect to the worker, fetch its hello, and validate the
    /// protocol version.
    async fn connect(&self) -> Result<(Arc<WorkerClient>, WorkerHello), WorkerSwitchError> {
        // Policy guard: remote (non-loopback) endpoints require TLS.
        // The tonic TLS connector only engages for the `https://`
        // scheme, so both TLS AND an https endpoint are required —
        // anything else would downgrade to cleartext.
        let https_scheme = self
            .config
            .endpoint
            .to_ascii_lowercase()
            .starts_with("https://");
        if !is_loopback_endpoint(&self.config.endpoint) && !https_scheme {
            return Err(WorkerSwitchError::Startup {
                message: format!(
                    "refusing cleartext connection to non-loopback endpoint `{}`: TLS (https:// endpoint) is required for remote workers",
                    self.config.endpoint
                ),
            });
        }

        let mut last_error = String::new();
        let attempts = self.config.connect_retries + 1;
        for attempt in 0..attempts {
            let attempt_connect = async {
                if https_scheme && self.config.auth.tls.is_none() {
                    // Default remote trust: TOFU. The first connect pins
                    // the presented certificate in the trust store;
                    // later connects require a match.
                    connect_with_trust(
                        &self.config.endpoint,
                        reimagine_backend_worker_protocol::remote::ConnectTrust::new(
                            self.config.trust_store.as_ref(),
                            &self.config.worker_id,
                        ),
                        &self.config.auth,
                    )
                    .await
                } else {
                    // Loopback dev endpoints, and https endpoints whose
                    // auth carries an explicit TLS mode (`GrpcTls`):
                    // the explicit trust configuration overrides TOFU.
                    WorkerClient::connect(&self.config.endpoint, &self.config.auth).await
                }
            };
            match tokio::time::timeout(self.config.connect_timeout, attempt_connect).await {
                Ok(Ok(client)) => {
                    let hello = self.hello(&client).await?;
                    return Ok((Arc::new(client), hello));
                }
                Ok(Err(error)) => {
                    last_error = format!("{error}");
                    tracing::warn!(
                        "gRPC connect to `{}` failed (attempt {}/{}): {error}",
                        self.config.endpoint,
                        attempt + 1,
                        attempts,
                    );
                }
                Err(_elapsed) => {
                    last_error =
                        format!("connect timed out after {:?}", self.config.connect_timeout);
                    tracing::warn!(
                        "gRPC connect to `{}` timed out (attempt {}/{})",
                        self.config.endpoint,
                        attempt + 1,
                        attempts,
                    );
                }
            }
        }

        Err(WorkerSwitchError::Startup {
            message: format!(
                "gRPC connect to `{}` failed after {} attempts: {last_error}",
                self.config.endpoint, attempts,
            ),
        })
    }

    /// Fetch and validate the worker's hello.
    async fn hello(&self, client: &WorkerClient) -> Result<WorkerHello, WorkerSwitchError> {
        let response = tokio::time::timeout(self.config.connect_timeout, client.hello())
            .await
            .map_err(|_elapsed| WorkerSwitchError::Startup {
                message: format!(
                    "waiting for hello from `{}` timed out after {:?}",
                    self.config.endpoint, self.config.connect_timeout,
                ),
            })?
            .map_err(|error| WorkerSwitchError::Startup {
                message: format!("worker hello failed: {error}"),
            })?;
        let hello = reimagine_backend_worker_protocol::conversion::hello_from_proto(response)
            .ok_or_else(|| WorkerSwitchError::Startup {
                message: format!(
                    "worker `{}` returned an incomplete hello",
                    self.config.endpoint
                ),
            })?;
        if hello.protocol.0 != reimagine_backend_worker_protocol::WORKER_PROTOCOL_VERSION.0 {
            return Err(WorkerSwitchError::Startup {
                message: format!(
                    "worker `{}` speaks protocol v{} but this host requires v{}; reinstall the worker",
                    self.config.endpoint,
                    hello.protocol.0,
                    reimagine_backend_worker_protocol::WORKER_PROTOCOL_VERSION.0,
                ),
            });
        }
        Ok(hello)
    }
}

#[async_trait]
impl WorkerSwitchTarget for RemoteWorkerCandidate {
    async fn start(&self) -> Result<Arc<dyn SwitchableWorker>, WorkerSwitchError> {
        let (client, hello) = self.connect().await?;
        Ok(Arc::new(RemoteSwitchableWorker::new(hello, client)))
    }
}

/// Whether an endpoint URI targets the loopback interface.
///
/// Loopback endpoints may run plain HTTP for local development; any
/// other endpoint requires TLS. Understands `http(s)://host:port`,
/// `[::1]` bracket forms, and bare hosts without a scheme.
fn is_loopback_endpoint(endpoint: &str) -> bool {
    let without_scheme = endpoint
        .strip_prefix("http://")
        .or_else(|| endpoint.strip_prefix("https://"))
        .unwrap_or(endpoint);
    let authority = without_scheme.split('/').next().unwrap_or(without_scheme);
    let host = if let Some(bracketed) = authority.strip_prefix('[') {
        bracketed.split(']').next().unwrap_or("")
    } else {
        // `host:port`, or a raw IPv6 literal. For raw IPv6 the final
        // `:port` component is digits; treat everything before it as
        // the host.
        let mut parts = authority.rsplitn(2, ':');
        let (maybe_port, rest) = (parts.next().unwrap_or(""), parts.next());
        if maybe_port.bytes().all(|b| b.is_ascii_digit()) {
            rest.unwrap_or(maybe_port)
        } else {
            authority.split(':').next().unwrap_or(authority)
        }
    };
    let host = host.to_ascii_lowercase();
    matches!(host.as_str(), "localhost" | "127.0.0.1" | "::1")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_with_defaults_is_generous() {
        let (_dir, store) = test_store();
        let config = RemoteWorkerCandidate::with_defaults(
            "worker-1",
            "https://127.0.0.1:59997",
            GrpcAuth::plain(),
            store,
        )
        .config;
        assert_eq!(config.connect_timeout, std::time::Duration::from_secs(10));
        assert_eq!(config.connect_retries, 2);
        assert!(config.auth.token.is_none());
        assert!(config.auth.tls.is_none());
    }

    #[test]
    fn candidate_config_debug_redacts_auth() {
        let (_dir, store) = test_store();
        let config = RemoteWorkerCandidateConfig {
            worker_id: "worker-1".to_owned(),
            endpoint: "https://worker.example:50051".to_owned(),
            auth: GrpcAuth {
                token: Some("s3cret".to_owned()),
                tls: Some(
                    reimagine_backend_worker_protocol::GrpcTls::InsecureSkipVerify {
                        domain: "worker.example".to_owned(),
                    },
                ),
            },
            trust_store: store,
            connect_timeout: std::time::Duration::from_secs(10),
            connect_retries: 2,
        };
        let rendered = format!("{config:?}");
        assert!(
            !rendered.contains("s3cret"),
            "token leaked in Debug: {rendered}"
        );
        assert!(rendered.contains("redacted"));
    }

    #[test]
    fn loopback_endpoint_detection() {
        let loopback = [
            "http://127.0.0.1:50051",
            "http://localhost:50051",
            "https://localhost:50051",
            "http://[::1]:50051",
            "http://::1:50051",
            "127.0.0.1:50051",
        ];
        for endpoint in loopback {
            assert!(
                is_loopback_endpoint(endpoint),
                "{endpoint} should be loopback"
            );
        }
        let remote = [
            "http://192.168.1.50:50051",
            "http://worker.example:50051",
            "https://worker.example:50051",
            "http://10.0.0.7:50051",
        ];
        for endpoint in remote {
            assert!(
                !is_loopback_endpoint(endpoint),
                "{endpoint} should not be loopback"
            );
        }
    }

    #[tokio::test]
    async fn candidate_refuses_cleartext_non_loopback_endpoint() {
        let (_dir, store) = test_store();
        let candidate = RemoteWorkerCandidate::with_defaults(
            "worker-1",
            "http://192.168.1.50:50051",
            GrpcAuth::plain(),
            store,
        );
        let error = match candidate.start().await {
            Ok(_) => panic!("must be refused"),
            Err(error) => error,
        };
        assert!(matches!(error, WorkerSwitchError::Startup { .. }));
        assert!(
            error.to_string().contains("cleartext"),
            "expected cleartext refusal, got: {error}"
        );
        assert!(error.to_string().contains("TLS"));
    }

    #[tokio::test]
    async fn candidate_with_tls_accepts_non_loopback_endpoint() {
        // TLS configured: the candidate guard passes; the failure (if
        // any) comes from the connect itself, not the policy guard.
        // `127.0.0.2` is loopback /8 but not the literal `127.0.0.1`,
        // so the guard treats it as remote while the connect fails fast
        // (nothing listens there) without DNS.
        let (_dir, store) = test_store();
        let candidate = RemoteWorkerCandidate::new(RemoteWorkerCandidateConfig {
            worker_id: "worker-1".to_owned(),
            endpoint: "https://127.0.0.2:50051".to_owned(),
            auth: GrpcAuth::plain(),
            trust_store: store,
            connect_timeout: std::time::Duration::from_secs(2),
            connect_retries: 0,
        });
        let error = match candidate.start().await {
            Ok(_) => panic!("connect fails"),
            Err(error) => error,
        };
        assert!(!error.to_string().contains("cleartext"), "got: {error}");
    }

    fn test_store() -> (tempfile::TempDir, Arc<Mutex<TrustedKeyStore>>) {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(Mutex::new(
            TrustedKeyStore::load(dir.path().join("trusted-keys.json")).unwrap(),
        ));
        (dir, store)
    }
}
