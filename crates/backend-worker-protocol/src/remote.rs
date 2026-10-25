//! TOFU certificate pinning for remote `https://` worker connections.
//!
//! A remote worker serves the v2 contract over TLS with a self-signed
//! certificate. On the first connect the presented certificate's
//! fingerprint (SHA-256 over the DER bytes, hex-encoded) is recorded in
//! the [`TrustedKeyStore`]; every later connect requires a match. Pins
//! are keyed by `worker_id@endpoint-host` so a worker that moves hosts
//! re-pins instead of silently reusing a stale pin, while port changes
//! (worker restarts) keep the pin.
//!
//! # Security note (auto-connect)
//! TOFU records the first certificate it sees for a key with no
//! out-of-band confirmation. mDNS is unauthenticated, so an attacker on
//! the LAN can advertise a spoofed worker and win the first-connect
//! race. Auto-connect flows MUST use [`ConnectTrust::require_pin`] with
//! an explicit user trust action for unknown workers; TOFU is only safe
//! for operator-initiated connects.

use std::sync::{Arc, Mutex};

use rustls::client::danger::ServerCertVerifier;
use tonic::Status;
use tonic::transport::{ClientTlsConfig, Endpoint};

use crate::auth::GrpcAuth;
use crate::client::WorkerClient;
use crate::tls::ensure_crypto_provider;
use crate::trust::{
    CertFingerprint, FingerprintServerVerifier, TrustedKeyStore, insecure_skip_pinning_from_env,
};

/// Whether the peer certificate must already be pinned before connecting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrustPolicy {
    /// Trust On First Use: record and pin the presented fingerprint on the
    /// first connect, then require a match on later connects.
    Tofu,
    /// Refuse to connect unless a pin for `worker_id` already exists.
    RequirePin,
}

/// Client-side trust configuration for a TOFU/pinning connect.
#[derive(Clone, Copy, Debug)]
pub struct ConnectTrust<'a> {
    store: &'a Mutex<TrustedKeyStore>,
    worker_id: &'a str,
    policy: TrustPolicy,
    skip_pinning: bool,
}

impl<'a> ConnectTrust<'a> {
    /// TOFU trust against `store`, keyed by `worker_id`.
    ///
    /// `worker_id` should be the stable endpoint/worker identifier the
    /// orchestrator uses for this worker (e.g. the mDNS instance id).
    #[must_use]
    pub fn new(store: &'a Mutex<TrustedKeyStore>, worker_id: &'a str) -> Self {
        Self {
            store,
            worker_id,
            policy: TrustPolicy::Tofu,
            skip_pinning: false,
        }
    }

    /// Refuse to connect unless a pin already exists (no recording).
    #[must_use]
    pub fn require_pin(mut self) -> Self {
        self.policy = TrustPolicy::RequirePin;
        self
    }

    /// Explicitly bypass pinning (dev/test escape hatch). The presented
    /// certificate is accepted without being recorded.
    #[must_use]
    pub fn insecure_skip(mut self) -> Self {
        self.skip_pinning = true;
        self
    }
}

/// Connect a [`WorkerClient`] to a remote `https://` worker with TOFU
/// certificate pinning.
///
/// On the first connect (TOFU) the presented certificate fingerprint is
/// recorded in the store keyed by `worker_id` and the endpoint host; on
/// later connects the presented fingerprint must match the recorded pin,
/// otherwise the handshake fails. `RequirePin` refuses to connect when
/// no pin exists.
///
/// Pinning is skipped entirely (accept-any, nothing recorded) when
/// [`ConnectTrust::insecure_skip`] is set or
/// `REIMAGINE_INSECURE_SKIP_PINNING` is truthy — a dev/test escape
/// hatch, default off.
pub async fn connect_with_trust(
    endpoint: &str,
    trust: ConnectTrust<'_>,
    auth: &GrpcAuth,
) -> Result<WorkerClient, Status> {
    if !endpoint.to_ascii_lowercase().starts_with("https://") {
        return Err(Status::invalid_argument(
            "certificate pinning requires an https:// endpoint",
        ));
    }
    let host = endpoint_host(endpoint)
        .ok_or_else(|| Status::invalid_argument(format!("invalid https endpoint `{endpoint}`")))?;
    // Composite key: worker identity + endpoint host. The host component
    // makes pins specific to the machine the orchestrator connected to
    // (a worker that moves hosts re-pins); port changes (restarts) do
    // not invalidate the pin.
    let pin_key = format!("{}@{}", trust.worker_id, host);

    let skip_pinning = trust.skip_pinning || insecure_skip_pinning_from_env();
    if skip_pinning && trust.skip_pinning {
        tracing::warn!(
            "certificate pinning bypassed (insecure_skip) for worker '{}'",
            trust.worker_id
        );
    }
    if skip_pinning && insecure_skip_pinning_from_env() {
        tracing::warn!("certificate pinning bypassed by REIMAGINE_INSECURE_SKIP_PINNING env");
    }

    let expected: Option<CertFingerprint> = if skip_pinning {
        None
    } else {
        let guard = trust
            .store
            .lock()
            .map_err(|e| Status::unavailable(format!("trust store lock poisoned: {e}")))?;
        match (guard.pinned(&pin_key), trust.policy) {
            (Some(fingerprint), _) => Some(fingerprint.clone()),
            (None, TrustPolicy::RequirePin) => {
                return Err(Status::unavailable(format!(
                    "no pinned certificate for worker '{}' (require_pin)",
                    trust.worker_id
                )));
            }
            (None, TrustPolicy::Tofu) => None,
        }
    };

    ensure_crypto_provider();
    let captured = Arc::new(Mutex::new(None));
    let verifier: Arc<dyn ServerCertVerifier> = Arc::new(FingerprintServerVerifier::new(
        expected.clone(),
        Arc::clone(&captured),
    ));
    let channel = Endpoint::from_shared(endpoint.to_owned())
        .map_err(|error| Status::unavailable(format!("invalid endpoint `{endpoint}`: {error}")))?
        .tls_config_with_verifier(ClientTlsConfig::new().domain_name(host), verifier)
        .map_err(|error| Status::unavailable(format!("invalid TLS config: {error}")))?
        .connect()
        .await
        .map_err(|error| Status::unavailable(format!("connect to `{endpoint}` failed: {error}")))?;

    // TOFU: record the presented fingerprint now that the handshake
    // succeeded. On pin-mismatch the handshake above already failed.
    //
    // The store lock is held across the record: a concurrent
    // first-connect that already recorded a *different* fingerprint
    // for this key must not be overwritten (last-writer-wins race).
    if expected.is_none() && !skip_pinning {
        let fingerprint = captured
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
            .ok_or_else(|| {
                Status::unavailable(
                    "handshake succeeded but no certificate fingerprint was captured",
                )
            })?;
        let mut guard = trust
            .store
            .lock()
            .map_err(|e| Status::unavailable(format!("trust store lock poisoned: {e}")))?;
        if let Some(existing) = guard.pinned(&pin_key)
            && existing != &fingerprint
        {
            return Err(Status::unavailable(format!(
                "worker '{}' presented fingerprint {} but {} is already pinned for {}",
                trust.worker_id,
                fingerprint.as_str(),
                existing.as_str(),
                pin_key
            )));
        }
        guard
            .trust(&pin_key, &fingerprint)
            .map_err(|e| Status::unavailable(format!("failed to pin certificate: {e}")))?;
    }

    Ok(WorkerClient::from_channel(
        channel,
        endpoint.to_owned(),
        auth,
    ))
}

/// The host component of an `https://` endpoint URI (the TOFU pin key's
/// address part): authority without scheme, userinfo, port, or brackets.
fn endpoint_host(endpoint: &str) -> Option<String> {
    let authority = endpoint
        .split_once("://")?
        .1
        .split(['/', '?', '#'])
        .next()?;
    let authority = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    let host = if let Some(bracketed) = authority.strip_prefix('[') {
        bracketed.split(']').next()?
    } else {
        let mut parts = authority.rsplitn(2, ':');
        let maybe_port = parts.next()?;
        match parts.next() {
            Some(rest) if maybe_port.bytes().all(|b| b.is_ascii_digit()) => rest,
            _ => authority,
        }
    };
    if host.is_empty() {
        None
    } else {
        Some(host.to_ascii_lowercase())
    }
}

#[cfg(test)]
mod tests {
    use super::endpoint_host;

    #[test]
    fn endpoint_host_extracts_the_pin_address_part() {
        assert_eq!(
            endpoint_host("https://192.168.1.10:9100").as_deref(),
            Some("192.168.1.10")
        );
        assert_eq!(
            endpoint_host("https://127.0.0.1:50051").as_deref(),
            Some("127.0.0.1")
        );
        assert_eq!(
            endpoint_host("https://worker.example:50051").as_deref(),
            Some("worker.example")
        );
        assert_eq!(endpoint_host("https://[::1]:50051").as_deref(), Some("::1"));
        assert_eq!(
            endpoint_host("https://worker.example").as_deref(),
            Some("worker.example")
        );
        assert_eq!(
            endpoint_host("https://User@host:443/x").as_deref(),
            Some("host")
        );
        assert_eq!(endpoint_host("unix:///tmp/worker.sock"), None);
        assert_eq!(endpoint_host("https://"), None);
    }
}
