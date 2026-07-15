//! Minimal TUF metadata verification.
//!
//! Implements a subset of the TUF specification sufficient for
//! authenticating worker packages:
//!
//! - Root metadata trust (embedded in the application binary)
//! - Timestamp → Snapshot → Targets chain verification
//! - Ed25519 signature verification via `ed25519-dalek`
//! - Expiry, freeze-attack, and rollback protection
//! - Consistent snapshot and version linkage
//! - Target length and hash verification

use std::collections::HashMap;
use std::time::SystemTime;

use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::CatalogError;

// ── TUF Key types ──────────────────────────────────────────────────

/// A TUF key (Ed25519).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TufKey {
    #[serde(rename = "keytype")]
    pub key_type: String,
    pub scheme: String,
    pub keyval: TufKeyVal,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TufKeyVal {
    pub public: String,
}

impl TufKey {
    /// Parse the hex-encoded Ed25519 public key into a verifying key.
    pub fn to_verifying_key(&self) -> Result<VerifyingKey, CatalogError> {
        let bytes = hex::decode(&self.keyval.public).map_err(|e| CatalogError::RootLoad {
            message: format!("invalid key hex: {e}"),
        })?;
        VerifyingKey::from_bytes(&bytes.try_into().map_err(|_| CatalogError::RootLoad {
            message: "invalid Ed25519 key length".to_string(),
        })?)
        .map_err(|e| CatalogError::RootLoad {
            message: format!("invalid Ed25519 key: {e}"),
        })
    }
}

// ── TUF Role types ─────────────────────────────────────────────────

/// Role configuration: which key IDs sign for this role and the
/// threshold of valid signatures required.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoleConfig {
    pub keyids: Vec<String>,
    pub threshold: usize,
}

// ── Root Metadata ──────────────────────────────────────────────────

/// TUF root metadata (v1 subset).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RootMetadata {
    pub signed: RootSigned,
    pub signatures: Vec<SignatureEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RootSigned {
    #[serde(rename = "_type")]
    pub kind: String,
    pub version: u64,
    pub expires: String,
    pub keys: HashMap<String, TufKey>,
    pub roles: HashMap<String, RoleConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SignatureEntry {
    pub keyid: String,
    pub sig: String,
}

// ── Timestamp Metadata ─────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimestampMetadata {
    pub signed: TimestampSigned,
    pub signatures: Vec<SignatureEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimestampSigned {
    #[serde(rename = "_type")]
    pub kind: String,
    pub version: u64,
    pub expires: String,
    pub meta: HashMap<String, SnapshotMetaEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SnapshotMetaEntry {
    pub version: u64,
    #[serde(default)]
    pub length: Option<u64>,
    #[serde(default)]
    pub hashes: Option<HashMap<String, String>>,
}

// ── Snapshot Metadata ──────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    pub signed: SnapshotSigned,
    pub signatures: Vec<SignatureEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SnapshotSigned {
    #[serde(rename = "_type")]
    pub kind: String,
    pub version: u64,
    pub expires: String,
    pub meta: HashMap<String, TargetsMetaEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TargetsMetaEntry {
    pub version: u64,
    #[serde(default)]
    pub length: Option<u64>,
    #[serde(default)]
    pub hashes: Option<HashMap<String, String>>,
}

// ── Targets Metadata ───────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TargetsMetadata {
    pub signed: TargetsSigned,
    pub signatures: Vec<SignatureEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TargetsSigned {
    #[serde(rename = "_type")]
    pub kind: String,
    pub version: u64,
    pub expires: String,
    pub targets: HashMap<String, TargetDesc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TargetDesc {
    pub length: u64,
    pub hashes: HashMap<String, String>,
    #[serde(default)]
    pub custom: Option<serde_json::Value>,
}

// ── Verification helpers ───────────────────────────────────────────

fn parse_expiry(expires: &str) -> Result<DateTime<Utc>, CatalogError> {
    expires
        .parse::<DateTime<Utc>>()
        .map_err(|e| CatalogError::Json {
            path: None,
            message: format!("invalid expiry timestamp `{expires}`: {e}"),
        })
}

fn check_expiry(role: &str, expires: &str) -> Result<(), CatalogError> {
    let expiry = parse_expiry(expires)?;
    if SystemTime::now() > expiry.into() {
        return Err(CatalogError::Expired {
            role: role.to_string(),
            expires: expires.to_string(),
        });
    }
    Ok(())
}

fn check_rollback(role: &str, stored: u64, attempted: u64) -> Result<(), CatalogError> {
    if attempted < stored {
        return Err(CatalogError::Rollback {
            role: role.to_string(),
            stored,
            attempted,
        });
    }
    Ok(())
}

/// Verify Ed25519 signatures in metadata against the provided keys.
fn verify_signatures(
    role: &str,
    signed_bytes: &[u8],
    signatures: &[SignatureEntry],
    keys: &HashMap<String, TufKey>,
    role_config: &RoleConfig,
) -> Result<(), CatalogError> {
    let mut valid_count = 0usize;

    for sig_entry in signatures {
        let Some(key) = keys.get(&sig_entry.keyid) else {
            // Unknown key — skip (per TUF spec, unknown keys are ignored)
            continue;
        };
        if !role_config.keyids.contains(&sig_entry.keyid) {
            // Key not authorized for this role — skip
            continue;
        }
        let verifying_key = key.to_verifying_key()?;
        let sig_bytes = hex::decode(&sig_entry.sig).map_err(|e| CatalogError::Signature {
            role: role.to_string(),
            key_id: sig_entry.keyid.clone(),
            message: format!("invalid signature hex: {e}"),
        })?;
        let signature = Signature::from_slice(&sig_bytes).map_err(|e| CatalogError::Signature {
            role: role.to_string(),
            key_id: sig_entry.keyid.clone(),
            message: format!("invalid signature: {e}"),
        })?;

        match verifying_key.verify(signed_bytes, &signature) {
            Ok(()) => valid_count += 1,
            Err(e) => {
                return Err(CatalogError::Signature {
                    role: role.to_string(),
                    key_id: sig_entry.keyid.clone(),
                    message: format!("verification failed: {e}"),
                });
            }
        }
    }

    if valid_count < role_config.threshold {
        return Err(CatalogError::ThresholdNotMet {
            role: role.to_string(),
            required: role_config.threshold,
            received: valid_count,
        });
    }

    Ok(())
}

/// Serialize a signed payload to canonical bytes for signature verification.
/// Uses a consistent serialization (lexicographic key order via JSON).
fn serialize_signed<T: Serialize>(signed: &T) -> Result<Vec<u8>, CatalogError> {
    let value = serde_json::to_value(signed).map_err(|e| CatalogError::Json {
        path: None,
        message: format!("serialization failed: {e}"),
    })?;
    serde_json::to_vec(&value).map_err(|e| CatalogError::Json {
        path: None,
        message: format!("canonical serialization failed: {e}"),
    })
}

fn role_config<'a>(root: &'a RootMetadata, role: &str) -> Result<&'a RoleConfig, CatalogError> {
    root.signed
        .roles
        .get(role)
        .ok_or_else(|| CatalogError::RootLoad {
            message: format!("root metadata has no {role} role config"),
        })
}

fn verify_role<T: Serialize>(
    role: &str,
    signed: &T,
    signatures: &[SignatureEntry],
    root: &RootMetadata,
) -> Result<(), CatalogError> {
    verify_signatures(
        role,
        &serialize_signed(signed)?,
        signatures,
        &root.signed.keys,
        role_config(root, role)?,
    )
}

fn verify_linkage(
    role: &str,
    bytes: &[u8],
    expected_version: u64,
    actual_version: u64,
    expected_length: Option<u64>,
    expected_hashes: Option<&HashMap<String, String>>,
) -> Result<(), CatalogError> {
    if actual_version != expected_version {
        return Err(CatalogError::MetadataVersionMismatch {
            role: role.to_owned(),
            expected: expected_version,
            actual: actual_version,
        });
    }
    if let Some(expected) = expected_length
        && bytes.len() as u64 != expected
    {
        return Err(CatalogError::MetadataLengthMismatch {
            role: role.to_owned(),
            expected,
            actual: bytes.len() as u64,
        });
    }
    if let Some(expected) = expected_hashes.and_then(|hashes| hashes.get("sha256")) {
        let actual = hex::encode(Sha256::digest(bytes));
        if &actual != expected {
            return Err(CatalogError::MetadataHashMismatch {
                role: role.to_owned(),
                algorithm: "sha256".to_owned(),
            });
        }
    }
    Ok(())
}

// ── High-level verification functions ──────────────────────────────

/// Verify root metadata and return the trusted keys and role config.
///
/// This is the trust-on-first-use (TOFU) anchor. The application
/// embeds the initial `root.json`, and subsequent root updates are
/// verified against the previous root.
pub fn verify_root(
    root: &RootMetadata,
    trusted_root: Option<&RootMetadata>,
) -> Result<HashMap<String, TufKey>, CatalogError> {
    if root.signed.kind != "root" {
        return Err(CatalogError::RootLoad {
            message: format!("expected root metadata, got `{}`", root.signed.kind),
        });
    }

    // Expiry check
    check_expiry("root", &root.signed.expires)?;

    // Version rollback check (if we have a stored root)
    if let Some(trusted) = trusted_root {
        check_rollback("root", trusted.signed.version, root.signed.version)?;
    }

    // Verify signatures against the trusted keys
    let signable = serialize_signed(&root.signed)?;
    if let Some(trusted) = trusted_root {
        if root.signed.version != trusted.signed.version + 1 {
            return Err(CatalogError::MetadataVersionMismatch {
                role: "root".to_owned(),
                expected: trusted.signed.version + 1,
                actual: root.signed.version,
            });
        }
        verify_signatures(
            "root",
            &signable,
            &root.signatures,
            &trusted.signed.keys,
            role_config(trusted, "root")?,
        )?;
    }
    verify_signatures(
        "root",
        &signable,
        &root.signatures,
        &root.signed.keys,
        role_config(root, "root")?,
    )?;

    Ok(root.signed.keys.clone())
}

/// Verify timestamp metadata against root keys.
pub fn verify_timestamp(
    timestamp: &TimestampMetadata,
    root: &RootMetadata,
    stored_version: u64,
) -> Result<(), CatalogError> {
    if timestamp.signed.kind != "timestamp" {
        return Err(CatalogError::Json {
            path: None,
            message: format!(
                "expected timestamp metadata, got `{}`",
                timestamp.signed.kind
            ),
        });
    }

    check_expiry("timestamp", &timestamp.signed.expires)?;
    check_rollback("timestamp", stored_version, timestamp.signed.version)?;
    if stored_version > 0 && timestamp.signed.version == stored_version {
        return Err(CatalogError::FreezeAttack {
            role: "timestamp".to_owned(),
        });
    }

    verify_role("timestamp", &timestamp.signed, &timestamp.signatures, root)?;

    Ok(())
}

/// Verify snapshot metadata against root keys and timestamp linkage.
pub fn verify_snapshot(
    snapshot: &SnapshotMetadata,
    bytes: &[u8],
    root: &RootMetadata,
    expected: &SnapshotMetaEntry,
) -> Result<(), CatalogError> {
    if snapshot.signed.kind != "snapshot" {
        return Err(CatalogError::Json {
            path: None,
            message: format!("expected snapshot metadata, got `{}`", snapshot.signed.kind),
        });
    }

    check_expiry("snapshot", &snapshot.signed.expires)?;
    verify_linkage(
        "snapshot",
        bytes,
        expected.version,
        snapshot.signed.version,
        expected.length,
        expected.hashes.as_ref(),
    )?;
    verify_role("snapshot", &snapshot.signed, &snapshot.signatures, root)?;

    Ok(())
}

/// Verify targets metadata against root keys and snapshot linkage.
pub fn verify_targets(
    targets: &TargetsMetadata,
    bytes: &[u8],
    root: &RootMetadata,
    expected: &TargetsMetaEntry,
) -> Result<(), CatalogError> {
    if targets.signed.kind != "targets" {
        return Err(CatalogError::Json {
            path: None,
            message: format!("expected targets metadata, got `{}`", targets.signed.kind),
        });
    }

    check_expiry("targets", &targets.signed.expires)?;
    verify_linkage(
        "targets",
        bytes,
        expected.version,
        targets.signed.version,
        expected.length,
        expected.hashes.as_ref(),
    )?;
    verify_role("targets", &targets.signed, &targets.signatures, root)?;

    Ok(())
}

/// Verify that downloaded content matches the target's hash and length.
pub fn verify_target_content(
    target_path: &str,
    data: &[u8],
    target: &TargetDesc,
) -> Result<(), CatalogError> {
    // Verify length
    if data.len() as u64 != target.length {
        return Err(CatalogError::TargetLengthMismatch {
            target: target_path.to_string(),
            expected: target.length,
            actual: data.len() as u64,
        });
    }

    // Verify SHA-256 hash
    if let Some(expected_sha256) = target.hashes.get("sha256") {
        let actual_hash = hex::encode(Sha256::digest(data));
        if actual_hash != *expected_sha256 {
            return Err(CatalogError::TargetHashMismatch {
                target: target_path.to_string(),
                algorithm: "sha256".to_string(),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic test key pair (public key only).
    #[allow(dead_code)]
    fn test_keypair() -> &'static str {
        // Ed25519 test vector public key (from RFC 8032)
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
    }

    /// Create a signing key from the corresponding known private key.
    fn test_signing_key() -> ed25519_dalek::SigningKey {
        // Ed25519 test vector secret key (from RFC 8032)
        let seed: [u8; 32] =
            hex::decode("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
                .expect("valid hex")
                .try_into()
                .expect("valid 32-byte key");
        ed25519_dalek::SigningKey::from_bytes(&seed)
    }

    fn sign(signing_key: &ed25519_dalek::SigningKey, data: &[u8]) -> String {
        use ed25519_dalek::Signer;
        let signature = signing_key.sign(data);
        hex::encode(signature.to_bytes())
    }

    fn make_root(signing_key: &ed25519_dalek::SigningKey, version: u64) -> RootMetadata {
        let verifying_key = signing_key.verifying_key();
        let key_id = hex::encode(sha2::Sha256::digest(
            serde_json::to_vec(&serde_json::json!({
                "keytype": "ed25519",
                "scheme": "ed25519",
                "keyval": {"public": hex::encode(verifying_key.to_bytes())}
            }))
            .unwrap(),
        ));

        let key = TufKey {
            key_type: "ed25519".to_string(),
            scheme: "ed25519".to_string(),
            keyval: TufKeyVal {
                public: hex::encode(verifying_key.to_bytes()),
            },
        };

        let signed = RootSigned {
            kind: "root".to_string(),
            version,
            expires: "2999-01-01T00:00:00Z".to_string(),
            keys: {
                let mut m = HashMap::new();
                m.insert(key_id.clone(), key);
                m
            },
            roles: {
                let mut m = HashMap::new();
                m.insert(
                    "root".to_string(),
                    RoleConfig {
                        keyids: vec![key_id.clone()],
                        threshold: 1,
                    },
                );
                m.insert(
                    "timestamp".to_string(),
                    RoleConfig {
                        keyids: vec![key_id.clone()],
                        threshold: 1,
                    },
                );
                m.insert(
                    "snapshot".to_string(),
                    RoleConfig {
                        keyids: vec![key_id.clone()],
                        threshold: 1,
                    },
                );
                m.insert(
                    "targets".to_string(),
                    RoleConfig {
                        keyids: vec![key_id],
                        threshold: 1,
                    },
                );
                m
            },
        };

        let signature = sign(signing_key, &serialize_signed(&signed).unwrap());

        RootMetadata {
            signatures: vec![SignatureEntry {
                keyid: hex::encode(sha2::Sha256::digest(
                    serde_json::to_vec(&serde_json::json!({
                        "keytype": "ed25519",
                        "scheme": "ed25519",
                        "keyval": {"public": hex::encode(verifying_key.to_bytes())}
                    }))
                    .unwrap(),
                )),
                sig: signature,
            }],
            signed,
        }
    }

    #[test]
    fn root_verification_accepts_valid_root() {
        let signing_key = test_signing_key();
        let root = make_root(&signing_key, 1);
        assert!(verify_root(&root, None).is_ok());
    }

    fn signed_timestamp(
        signing_key: &ed25519_dalek::SigningKey,
        root: &RootMetadata,
        version: u64,
    ) -> TimestampMetadata {
        let signed = TimestampSigned {
            kind: "timestamp".to_owned(),
            version,
            expires: "2999-01-01T00:00:00Z".to_owned(),
            meta: HashMap::from([(
                "snapshot.json".to_owned(),
                SnapshotMetaEntry {
                    version: 4,
                    length: None,
                    hashes: None,
                },
            )]),
        };
        let keyid = root.signed.roles["timestamp"].keyids[0].clone();
        TimestampMetadata {
            signatures: vec![SignatureEntry {
                keyid,
                sig: sign(signing_key, &serialize_signed(&signed).unwrap()),
            }],
            signed,
        }
    }

    #[test]
    fn timestamp_uses_root_role_keys_and_threshold() {
        let signing_key = test_signing_key();
        let root = make_root(&signing_key, 1);
        let timestamp = signed_timestamp(&signing_key, &root, 2);

        assert!(verify_timestamp(&timestamp, &root, 1).is_ok());

        let unsigned = TimestampMetadata {
            signatures: Vec::new(),
            signed: timestamp.signed,
        };
        assert!(matches!(
            verify_timestamp(&unsigned, &root, 1),
            Err(CatalogError::ThresholdNotMet { .. })
        ));
    }

    #[test]
    fn unchanged_timestamp_version_is_a_freeze_attack() {
        let signing_key = test_signing_key();
        let root = make_root(&signing_key, 1);
        let timestamp = signed_timestamp(&signing_key, &root, 2);
        assert!(matches!(
            verify_timestamp(&timestamp, &root, 2),
            Err(CatalogError::FreezeAttack { .. })
        ));
    }

    #[test]
    fn snapshot_linkage_rejects_wrong_version_before_trust() {
        let signing_key = test_signing_key();
        let root = make_root(&signing_key, 1);
        let snapshot = SnapshotMetadata {
            signed: SnapshotSigned {
                kind: "snapshot".to_owned(),
                version: 3,
                expires: "2999-01-01T00:00:00Z".to_owned(),
                meta: HashMap::new(),
            },
            signatures: Vec::new(),
        };
        let bytes = serde_json::to_vec(&snapshot).unwrap();
        let expected = SnapshotMetaEntry {
            version: 4,
            length: Some(bytes.len() as u64),
            hashes: Some(HashMap::from([(
                "sha256".to_owned(),
                hex::encode(Sha256::digest(&bytes)),
            )])),
        };
        assert!(matches!(
            verify_snapshot(&snapshot, &bytes, &root, &expected),
            Err(CatalogError::MetadataVersionMismatch { .. })
        ));
    }

    #[test]
    fn expired_metadata_is_rejected() {
        let signed = RootSigned {
            kind: "root".to_string(),
            version: 1,
            expires: "2020-01-01T00:00:00Z".to_string(),
            keys: HashMap::new(),
            roles: HashMap::new(),
        };
        // Serialize to get canonical bytes
        let _bytes = serde_json::to_vec(&signed).unwrap();

        // Modify after serialization to create expired state
        let result = check_expiry("root", &signed.expires);
        assert!(result.is_err());
        match result.unwrap_err() {
            CatalogError::Expired { role, .. } => assert_eq!(role, "root"),
            _ => panic!("expected Expired error"),
        }
    }

    #[test]
    fn rollback_detected() {
        assert!(check_rollback("root", 2, 1).is_err());
        assert!(check_rollback("root", 2, 2).is_ok());
        assert!(check_rollback("root", 2, 3).is_ok());
    }

    #[test]
    fn target_hash_verification_works() {
        let data = b"hello world";
        let hash = hex::encode(Sha256::digest(data));
        let target = TargetDesc {
            length: data.len() as u64,
            hashes: {
                let mut m = HashMap::new();
                m.insert("sha256".to_string(), hash);
                m
            },
            custom: None,
        };

        assert!(verify_target_content("test.bin", data, &target).is_ok());
    }

    #[test]
    fn target_hash_mismatch_rejected() {
        let data = b"hello world";
        let target = TargetDesc {
            length: data.len() as u64,
            hashes: {
                let mut m = HashMap::new();
                m.insert(
                    "sha256".to_string(),
                    "0000000000000000000000000000000000000000000000000000deadbeef0000".to_string(),
                );
                m
            },
            custom: None,
        };

        assert!(verify_target_content("test.bin", data, &target).is_err());
    }

    #[test]
    fn target_length_mismatch_rejected() {
        let data = b"hello world";
        let hash = hex::encode(Sha256::digest(data));
        let target = TargetDesc {
            length: 999, // wrong
            hashes: {
                let mut m = HashMap::new();
                m.insert("sha256".to_string(), hash);
                m
            },
            custom: None,
        };

        assert!(verify_target_content("test.bin", data, &target).is_err());
    }
}
