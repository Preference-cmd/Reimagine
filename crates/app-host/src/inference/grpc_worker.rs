use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use reimagine_backend_worker_host::{WorkerProcessState, WorkerRunLeases};
use reimagine_backend_worker_protocol::{
    HostHello, ProtocolRange, WireMessage, WorkerHello, WorkerIncarnationId, WorkerTransport,
    negotiate_protocol,
};
use reimagine_backend_worker_transport_grpc::{
    GrpcAuth,
    client::{self},
    proto,
    transport::GrpcTransport,
};
use reimagine_inference::{
    Backend, BackendInstance, BackendInstanceSnapshot, DeviceProfile, InferenceBackend,
};

use super::switch::{SwitchableWorker, WorkerSwitchError, WorkerSwitchTarget};

/// A worker connected via gRPC, implementing [`SwitchableWorker`].
///
/// This wraps a gRPC-connected remote worker and provides the same
/// interface as [`ProcessSwitchableWorker`] for use with
/// [`WorkerSwitchService`].
pub struct GrpcSwitchableWorker {
    instance: BackendInstance,
    incarnation_id: WorkerIncarnationId,
    run_leases: Arc<WorkerRunLeases>,
    device_label: String,
    hello: WorkerHello,
    transport: Arc<GrpcTransport>,
}

impl GrpcSwitchableWorker {
    /// Create a new `GrpcSwitchableWorker` from a handshake result.
    ///
    /// The device label is taken from the worker's advertised instance
    /// profile (its own `WorkerHello`), falling back to `"remote"`.
    pub fn new(hello: WorkerHello, transport: Arc<GrpcTransport>) -> Self {
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
            transport,
        }
    }

    /// Access the underlying gRPC transport.
    pub fn transport(&self) -> &Arc<GrpcTransport> {
        &self.transport
    }
}

#[async_trait]
impl SwitchableWorker for GrpcSwitchableWorker {
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
        // gRPC workers are always ready once connected
        WorkerProcessState::Ready
    }

    fn inference_backend(&self) -> Option<Arc<dyn InferenceBackend>> {
        None // gRPC workers don't expose a local inference backend
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
                (
                    "remote_addr".to_owned(),
                    self.transport.description().endpoint.clone(),
                ),
            ]),
            diagnostics: Vec::new(),
        }
    }

    async fn shutdown(&self) -> Result<(), WorkerSwitchError> {
        self.transport
            .shutdown()
            .await
            .map_err(|error| WorkerSwitchError::Shutdown {
                instance: self.instance.clone(),
                message: error.to_string(),
            })
    }
}

/// Configuration for connecting to a remote gRPC worker.
#[derive(Clone)]
pub struct GrpcWorkerCandidateConfig {
    /// The endpoint URI, e.g. `http://127.0.0.1:50051` (loopback dev) or
    /// `https://worker.example:50051` (cloud; requires `auth.tls`).
    pub endpoint: String,
    /// TLS + bearer-token settings (T19). A token without TLS is refused
    /// by the client; non-loopback endpoints without TLS are refused by
    /// the candidate.
    pub auth: GrpcAuth,
    /// Per-attempt connect timeout. Generous for cloud reachability;
    /// this is NOT stdio's `startup_timeout` — it bounds a single
    /// network connect + handshake attempt.
    pub connect_timeout: Duration,
    /// Additional connect attempts after the first failure (0 = single
    /// attempt). Retries re-run the full connect + stream open.
    pub connect_retries: u32,
}

impl Default for GrpcWorkerCandidateConfig {
    fn default() -> Self {
        Self {
            endpoint: String::new(),
            auth: GrpcAuth::plain(),
            connect_timeout: Duration::from_secs(10),
            connect_retries: 2,
        }
    }
}

impl std::fmt::Debug for GrpcWorkerCandidateConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrpcWorkerCandidateConfig")
            .field("endpoint", &self.endpoint)
            .field("auth", &"<GrpcAuth (redacted)>")
            .field("connect_timeout", &self.connect_timeout)
            .field("connect_retries", &self.connect_retries)
            .finish()
    }
}

/// A candidate that connects to a remote worker via gRPC.
///
/// Implements [`WorkerSwitchTarget`] to enable switching from a local
/// stdio worker to a remote gRPC worker (Cloud GPU).
pub struct GrpcWorkerCandidate {
    config: GrpcWorkerCandidateConfig,
}

impl GrpcWorkerCandidate {
    /// Create a new candidate from a configuration.
    pub fn new(config: GrpcWorkerCandidateConfig) -> Self {
        Self { config }
    }

    /// Create a new candidate with default timeout/retry settings.
    pub fn with_defaults(endpoint: impl Into<String>, auth: GrpcAuth) -> Self {
        Self {
            config: GrpcWorkerCandidateConfig {
                endpoint: endpoint.into(),
                auth,
                ..GrpcWorkerCandidateConfig::default()
            },
        }
    }

    /// Connect to the worker and perform the gRPC handshake
    /// (`HostHello` -> `WorkerHello`), validating protocol agreement.
    async fn connect(&self) -> Result<(Arc<GrpcTransport>, WorkerHello), WorkerSwitchError> {
        if self.config.auth.tls.is_none() && !is_loopback_endpoint(&self.config.endpoint) {
            return Err(WorkerSwitchError::Startup {
                message: format!(
                    "refusing cleartext gRPC connection to non-loopback endpoint `{}`: TLS is required for remote (cloud) workers",
                    self.config.endpoint
                ),
            });
        }

        let mut last_error = String::new();
        let attempts = self.config.connect_retries + 1;
        for attempt in 0..attempts {
            match tokio::time::timeout(
                self.config.connect_timeout,
                client::connect_with(&self.config.endpoint, &self.config.auth),
            )
            .await
            {
                Ok(Ok(transport)) => {
                    let hello = self.handshake(&transport).await?;
                    return Ok((Arc::new(transport), hello));
                }
                Ok(Err(error)) => {
                    last_error = format!("{error}");
                    tracing::warn!(
                        "[app-host] gRPC connect to `{}` failed (attempt {}/{}): {error}",
                        self.config.endpoint,
                        attempt + 1,
                        attempts,
                    );
                }
                Err(_elapsed) => {
                    last_error =
                        format!("connect timed out after {:?}", self.config.connect_timeout);
                    tracing::warn!(
                        "[app-host] gRPC connect to `{}` timed out (attempt {}/{})",
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

    /// Exchange the `HostHello`/`WorkerHello` handshake over the opened
    /// communication stream and validate the negotiated protocol.
    async fn handshake(&self, transport: &GrpcTransport) -> Result<WorkerHello, WorkerSwitchError> {
        let host_hello = WireMessage::HostHello(HostHello {
            supported_protocols: ProtocolRange::new(1, 1),
        });
        let proto_message: proto::HostToWorker =
            (&host_hello)
                .try_into()
                .map_err(|error: String| WorkerSwitchError::Startup {
                    message: format!("serialize HostHello: {error}"),
                })?;
        transport
            .send(proto_message)
            .await
            .map_err(|error| WorkerSwitchError::Startup {
                message: format!("send HostHello: {error}"),
            })?;

        let response = tokio::time::timeout(self.config.connect_timeout, transport.recv())
            .await
            .map_err(|_elapsed| WorkerSwitchError::Startup {
                message: format!(
                    "waiting for WorkerHello from `{}` timed out after {:?}",
                    self.config.endpoint, self.config.connect_timeout,
                ),
            })?
            .map_err(|error| WorkerSwitchError::Startup {
                message: format!("recv WorkerHello: {error}"),
            })?
            .ok_or_else(|| WorkerSwitchError::Startup {
                message: format!(
                    "gRPC stream from `{}` closed before WorkerHello",
                    self.config.endpoint
                ),
            })?;

        let message = WireMessage::try_from(response).map_err(|error: String| {
            WorkerSwitchError::Startup {
                message: format!("deserialize WorkerHello: {error}"),
            }
        })?;
        let hello = match message {
            WireMessage::WorkerHello(hello) => hello,
            other => {
                return Err(WorkerSwitchError::Startup {
                    message: format!("expected WorkerHello, got {:?}", other.kind()),
                });
            }
        };

        // The worker selected a protocol; it must lie inside our
        // supported range (1..=1).
        let negotiated = negotiate_protocol(
            ProtocolRange::new(1, 1),
            ProtocolRange::new(hello.selected_protocol.0, hello.selected_protocol.0),
        )
        .map_err(|error| WorkerSwitchError::Startup {
            message: format!(
                "worker `{}` selected incompatible protocol: {error}",
                self.config.endpoint
            ),
        })?;
        tracing::debug!(
            "[app-host] gRPC handshake with `{}` negotiated protocol v{}",
            self.config.endpoint,
            negotiated.0,
        );

        Ok(hello)
    }
}

#[async_trait]
impl WorkerSwitchTarget for GrpcWorkerCandidate {
    async fn start(&self) -> Result<Arc<dyn SwitchableWorker>, WorkerSwitchError> {
        let (transport, hello) = self.connect().await?;
        Ok(Arc::new(GrpcSwitchableWorker::new(hello, transport)))
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
    use reimagine_backend_worker_transport_grpc::GrpcTls;

    #[test]
    fn candidate_config_defaults_are_generous() {
        let config = GrpcWorkerCandidateConfig::default();
        assert_eq!(config.connect_timeout, Duration::from_secs(10));
        assert_eq!(config.connect_retries, 2);
        assert!(config.auth.token.is_none());
        assert!(config.auth.tls.is_none());
    }

    #[test]
    fn candidate_config_debug_redacts_auth() {
        let config = GrpcWorkerCandidateConfig {
            endpoint: "https://worker.example:50051".to_owned(),
            auth: GrpcAuth {
                token: Some("s3cret".to_owned()),
                tls: Some(GrpcTls::InsecureSkipVerify {
                    domain: "worker.example".to_owned(),
                }),
            },
            ..GrpcWorkerCandidateConfig::default()
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
        let candidate =
            GrpcWorkerCandidate::with_defaults("http://192.168.1.50:50051", GrpcAuth::plain());
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
        let candidate = GrpcWorkerCandidate::new(GrpcWorkerCandidateConfig {
            endpoint: "https://127.0.0.2:50051".to_owned(),
            auth: GrpcAuth {
                token: None,
                tls: Some(GrpcTls::InsecureSkipVerify {
                    domain: "127.0.0.2".to_owned(),
                }),
            },
            connect_retries: 0,
            ..GrpcWorkerCandidateConfig::default()
        });
        let error = match candidate.start().await {
            Ok(_) => panic!("connect fails"),
            Err(error) => error,
        };
        assert!(!error.to_string().contains("cleartext"), "got: {error}");
    }
}
