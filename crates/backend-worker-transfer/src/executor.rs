//! Execution of [`TransferPlan`]s — the byte-movement half of T15/T17.
//!
//! Token-reference transfer first: [`TransferPlan::Local`] moves no bytes
//! (the key is already resolvable on the target side). `Ipc` and `Network`
//! plans push the payload through a [`TransferChannel`] and verify it came
//! back intact (length + checksum) before returning the bytes.
//!
//! TODO(T17 wiring): once T16's `TopologyAwareBridgePolicy` lands, this
//! executor must be wired into the bridge policy so cross-worker plans
//! execute as part of normal request dispatch.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use sha2::{Digest, Sha256};

use crate::TransferPlan;

/// Bytes of the integrity envelope header: magic (4) + length (8) + digest (32).
pub const ENVELOPE_HEADER_LEN: usize = 4 + 8 + 32;

const ENVELOPE_MAGIC: &[u8; 4] = b"TNSR";

/// A byte-level pipe to a worker.
///
/// Implementations map onto a concrete transport (QUIC bi-stream, gRPC
/// stream, in-memory shared buffer). The executor is transport-agnostic:
/// it only sees this trait plus the envelope format in
/// [`seal_envelope`]/[`open_envelope`].
#[async_trait]
pub trait TransferChannel: Send + Sync {
    /// Id of the worker this channel delivers to (matches the `target` of
    /// [`TransferPlan::Ipc`] / [`TransferPlan::Network`]).
    fn target_id(&self) -> &str;

    /// Push `bytes` to the target worker.
    async fn send(&self, bytes: &[u8]) -> Result<(), String>;

    /// Pull the target worker's stored bytes back.
    async fn recv(&self) -> Result<Vec<u8>, String>;
}

/// Executes [`TransferPlan`]s by moving tensor bytes between workers.
///
/// Channels are keyed by target worker id; [`execute`](Self::execute) routes
/// the payload through the channel of `plan.target`.
pub struct TransferExecutor {
    channels: HashMap<String, Arc<dyn TransferChannel>>,
}

impl std::fmt::Debug for TransferExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut keys: Vec<&str> = self.channels.keys().map(String::as_str).collect();
        keys.sort_unstable();
        f.debug_struct("TransferExecutor")
            .field("channels", &keys)
            .finish()
    }
}

impl Default for TransferExecutor {
    fn default() -> Self {
        Self::new(HashMap::new())
    }
}

impl TransferExecutor {
    /// Create an executor over per-target worker channels.
    pub fn new(channels: HashMap<String, Arc<dyn TransferChannel>>) -> Self {
        Self { channels }
    }

    /// The channel registered for `worker_id`, if any.
    pub fn channel(&self, worker_id: &str) -> Option<&Arc<dyn TransferChannel>> {
        self.channels.get(worker_id)
    }

    /// Move `payload` according to `plan`, returning the verified bytes.
    ///
    /// * [`TransferPlan::Local`] — no bytes move (token-reference path); the
    ///   payload is returned unchanged.
    /// * [`TransferPlan::Ipc`] / [`TransferPlan::Network`] — the payload is
    ///   sealed into an integrity envelope, transmitted via the target's
    ///   channel, and verified (length + checksum) on the echo before being
    ///   returned.
    pub async fn execute(&self, plan: &TransferPlan, payload: &[u8]) -> Result<Vec<u8>, String> {
        match plan {
            TransferPlan::Local { key } => {
                tracing::debug!(
                    key = %key,
                    bytes = payload.len(),
                    "local plan: token reference, no bytes moved"
                );
                Ok(payload.to_vec())
            }
            TransferPlan::Ipc { key, target } => self.transmit(key, target, payload).await,
            TransferPlan::Network {
                key,
                target,
                estimated_bytes,
            } => {
                if *estimated_bytes != payload.len() as u64 {
                    tracing::warn!(
                        key = %key,
                        estimated_bytes,
                        actual = payload.len(),
                        "plan estimated size differs from payload"
                    );
                }
                self.transmit(key, target, payload).await
            }
            TransferPlan::ObjectStorage { key, url } => Err(format!(
                "object-storage transfer not implemented in the executor (key `{key}`, url `{url}`)"
            )),
            TransferPlan::Unsupported { reason } => Err(reason.clone()),
        }
    }

    async fn transmit(&self, key: &str, target: &str, payload: &[u8]) -> Result<Vec<u8>, String> {
        let channel = self.channels.get(target).ok_or_else(|| {
            format!("no transfer channel registered for target worker `{target}`")
        })?;
        if channel.target_id() != target {
            return Err(format!(
                "channel keyed as `{target}` identifies itself as `{}`",
                channel.target_id()
            ));
        }
        let envelope = seal_envelope(payload);
        channel
            .send(&envelope)
            .await
            .map_err(|e| format!("send to `{target}` failed: {e}"))?;
        let echo = channel
            .recv()
            .await
            .map_err(|e| format!("recv from `{target}` failed: {e}"))?;
        open_envelope(&echo, payload.len())
            .map_err(|e| format!("integrity check for `{key}` failed: {e}"))
    }
}

/// Seal `payload` into an integrity envelope: magic + big-endian length +
/// SHA-256 digest + payload.
///
/// This is the byte format the executor sends through a
/// [`TransferChannel`]; the target stores it opaquely and echoes it back.
#[must_use]
pub fn seal_envelope(payload: &[u8]) -> Vec<u8> {
    let mut envelope = Vec::with_capacity(ENVELOPE_HEADER_LEN + payload.len());
    envelope.extend_from_slice(ENVELOPE_MAGIC);
    envelope.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    envelope.extend_from_slice(&Sha256::digest(payload));
    envelope.extend_from_slice(payload);
    envelope
}

/// Open an echoed envelope, verifying magic, declared length, and checksum
/// against `expected_len`. Returns the payload bytes.
pub fn open_envelope(echo: &[u8], expected_len: usize) -> Result<Vec<u8>, String> {
    if echo.len() != ENVELOPE_HEADER_LEN + expected_len {
        return Err(format!(
            "envelope length mismatch: expected {}, got {}",
            ENVELOPE_HEADER_LEN + expected_len,
            echo.len()
        ));
    }
    if echo[..4] != *ENVELOPE_MAGIC {
        return Err("envelope magic mismatch".to_owned());
    }
    let declared = u64::from_be_bytes(
        echo[4..12]
            .try_into()
            .map_err(|_| "truncated envelope header".to_owned())?,
    ) as usize;
    if declared != expected_len {
        return Err(format!(
            "envelope declares {declared} bytes, expected {expected_len}"
        ));
    }
    let digest = Sha256::digest(&echo[ENVELOPE_HEADER_LEN..]);
    if digest[..] != echo[12..ENVELOPE_HEADER_LEN] {
        return Err("envelope checksum mismatch".to_owned());
    }
    Ok(echo[ENVELOPE_HEADER_LEN..].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn patterned(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i.wrapping_mul(31) % 251) as u8).collect()
    }

    /// In-memory channel: stores what `send` pushes, returns it on `recv`.
    struct MemoryChannel {
        target_id: String,
        stored: Mutex<Vec<u8>>,
    }

    #[async_trait]
    impl TransferChannel for MemoryChannel {
        fn target_id(&self) -> &str {
            &self.target_id
        }

        async fn send(&self, bytes: &[u8]) -> Result<(), String> {
            let mut stored = self
                .stored
                .lock()
                .map_err(|_| "stored lock poisoned".to_owned())?;
            *stored = bytes.to_vec();
            Ok(())
        }

        async fn recv(&self) -> Result<Vec<u8>, String> {
            let stored = self
                .stored
                .lock()
                .map_err(|_| "stored lock poisoned".to_owned())?;
            Ok(stored.clone())
        }
    }

    fn executor_with(target: &str) -> (TransferExecutor, Arc<MemoryChannel>) {
        let channel = Arc::new(MemoryChannel {
            target_id: target.to_owned(),
            stored: Mutex::new(Vec::new()),
        });
        let executor = TransferExecutor::new(HashMap::from([(
            target.to_owned(),
            channel.clone() as Arc<dyn TransferChannel>,
        )]));
        (executor, channel)
    }

    #[test]
    fn envelope_roundtrip_preserves_payload() {
        let payload = patterned(64 * 1024);
        let envelope = seal_envelope(&payload);
        assert_eq!(envelope.len(), ENVELOPE_HEADER_LEN + payload.len());
        assert_eq!(open_envelope(&envelope, payload.len()).unwrap(), payload);
    }

    #[test]
    fn envelope_roundtrip_empty_payload() {
        let envelope = seal_envelope(&[]);
        assert_eq!(open_envelope(&envelope, 0).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn envelope_open_rejects_bitflip() {
        let payload = patterned(1024);
        let mut envelope = seal_envelope(&payload);
        let flip = envelope.len() / 2;
        envelope[flip] ^= 0x01;
        assert!(
            open_envelope(&envelope, payload.len())
                .unwrap_err()
                .contains("checksum")
        );
    }

    #[test]
    fn envelope_open_rejects_wrong_length() {
        let payload = patterned(1024);
        let envelope = seal_envelope(&payload);
        assert!(
            open_envelope(&envelope, 2048)
                .unwrap_err()
                .contains("length mismatch")
        );
    }

    #[test]
    fn envelope_open_rejects_bad_magic() {
        let payload = patterned(1024);
        let mut envelope = seal_envelope(&payload);
        envelope[0] = b'X';
        assert!(
            open_envelope(&envelope, payload.len())
                .unwrap_err()
                .contains("magic")
        );
    }

    #[tokio::test]
    async fn local_plan_moves_no_bytes() {
        let executor = TransferExecutor::new(HashMap::new());
        let payload = patterned(256);
        let plan = TransferPlan::Local {
            key: "a:a".to_owned(),
        };
        let moved = executor.execute(&plan, &payload).await.unwrap();
        assert_eq!(moved, payload);
    }

    #[tokio::test]
    async fn ipc_and_network_plans_roundtrip_through_channel() {
        let (executor, channel) = executor_with("worker-b");
        let payload = patterned(128 * 1024);
        for plan in [
            TransferPlan::Ipc {
                key: "a:b".to_owned(),
                target: "worker-b".to_owned(),
            },
            TransferPlan::Network {
                key: "a:b".to_owned(),
                target: "worker-b".to_owned(),
                estimated_bytes: payload.len() as u64,
            },
        ] {
            let moved = executor.execute(&plan, &payload).await.unwrap();
            assert_eq!(moved, payload);
            let stored = channel.stored.lock().unwrap();
            assert_eq!(&*stored, &seal_envelope(&payload));
        }
    }

    #[tokio::test]
    async fn network_plan_reports_size_mismatch_but_transfers() {
        let (executor, _channel) = executor_with("worker-b");
        let payload = patterned(1024);
        let plan = TransferPlan::Network {
            key: "a:b".to_owned(),
            target: "worker-b".to_owned(),
            estimated_bytes: 9999,
        };
        let moved = executor.execute(&plan, &payload).await.unwrap();
        assert_eq!(moved, payload);
    }

    #[tokio::test]
    async fn network_plan_without_channel_errors() {
        let executor = TransferExecutor::new(HashMap::new());
        let plan = TransferPlan::Network {
            key: "a:b".to_owned(),
            target: "worker-b".to_owned(),
            estimated_bytes: 1,
        };
        let err = executor.execute(&plan, &[0u8]).await.unwrap_err();
        assert!(err.contains("worker-b"), "{err}");
    }

    #[tokio::test]
    async fn channel_with_mismatched_identity_errors() {
        let channel: Arc<dyn TransferChannel> = Arc::new(MemoryChannel {
            target_id: "other".to_owned(),
            stored: Mutex::new(Vec::new()),
        });
        let executor = TransferExecutor::new(HashMap::from([("worker-b".to_owned(), channel)]));
        let plan = TransferPlan::Network {
            key: "a:b".to_owned(),
            target: "worker-b".to_owned(),
            estimated_bytes: 1,
        };
        let err = executor.execute(&plan, &[0u8]).await.unwrap_err();
        assert!(err.contains("identifies itself as"), "{err}");
    }

    #[tokio::test]
    async fn unsupported_and_object_storage_plans_error() {
        let executor = TransferExecutor::new(HashMap::new());
        let unsupported = TransferPlan::Unsupported {
            reason: "no route".to_owned(),
        };
        assert_eq!(
            executor.execute(&unsupported, &[0u8]).await.unwrap_err(),
            "no route"
        );
        let object_storage = TransferPlan::ObjectStorage {
            key: "a:b".to_owned(),
            url: "s3://bucket/key".to_owned(),
        };
        let err = executor.execute(&object_storage, &[0u8]).await.unwrap_err();
        assert!(err.contains("object-storage"), "{err}");
    }
}
