//! Durable TUF trust state for the worker catalog.
//!
//! Persists the highest trusted root and the highest accepted timestamp
//! version under the application-global worker store. Used by
//! [`CatalogClient`](super::CatalogClient) so that restart preserves
//! rollback and freeze-attack protection instead of resetting to zero.
//!
//! Implementation notes:
//! - Storage is atomic: write to a sibling temp file, `sync_all`, then
//!   `rename`. A crash mid-write must never leave a partial file behind.
//! - Loads are fail-closed: a corrupt or missing state file does NOT
//!   silently reset to `timestamp_version = 0`, because that would
//!   disable rollback protection. Callers must bootstrap a fresh
//!   state via [`TrustedCatalogState::bootstrap_from_embedded_root`]
//!   explicitly when no state exists yet.
//! - The root keys in state must equal the keys derived from the stored
//!   root; a mismatch is treated as tamper and rejected.

use std::collections::HashMap;
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::error::CatalogError;
use super::tuf::{RootMetadata, TufKey};

/// In-memory representation of durable TUF trust state.
///
/// Holds the embedded/loaded root metadata, the set of keys authorized by
/// that root, and the highest accepted timestamp version.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrustedCatalogState {
    /// The trusted root metadata.
    pub root: RootMetadata,
    /// Key IDs and public keys authorized by the root.
    pub root_keys: HashMap<String, TufKey>,
    /// Highest accepted timestamp version seen in this run or persisted
    /// from a previous run. Starts at `0` on first bootstrap.
    pub stored_timestamp_version: u64,
}

/// Errors that can occur while persisting or restoring trust state.
#[derive(Debug)]
pub enum TrustedStateError {
    Io {
        path: PathBuf,
        message: String,
    },
    Parse {
        path: PathBuf,
        message: String,
    },
    RootKeysMismatch {
        missing: Vec<String>,
        unexpected: Vec<String>,
    },
    RootVerification {
        message: String,
    },
}

impl fmt::Display for TrustedStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, message } => {
                write!(f, "I/O error on `{}`: {}", path.display(), message)
            }
            Self::Parse { path, message } => {
                write!(f, "failed to parse trusted state at `{}`: {}", path.display(), message)
            }
            Self::RootKeysMismatch { missing, unexpected } => {
                write!(
                    f,
                    "stored root keys disagree with derived keys from root metadata: \
                     missing from derived = {missing:?}, unexpected in stored = {unexpected:?}"
                )
            }
            Self::RootVerification { message } => {
                write!(f, "root metadata failed verification: {message}")
            }
        }
    }
}

impl std::error::Error for TrustedStateError {}

impl TrustedCatalogState {
    /// Bootstrap a fresh state from a pre-verified root metadata and its
    /// derived key set. Used both on first startup (no prior state) and
    /// during root rotation after the new root has been verified.
    #[must_use]
    pub fn bootstrap_from_root(root: RootMetadata, root_keys: HashMap<String, TufKey>) -> Self {
        Self {
            root,
            root_keys,
            stored_timestamp_version: 0,
        }
    }

    /// Advance the stored timestamp version if `fetched` is strictly greater
    /// than the current value. Returns `true` if the state was advanced.
    pub fn record_timestamp_version(&mut self, fetched: u64) -> bool {
        if fetched > self.stored_timestamp_version {
            self.stored_timestamp_version = fetched;
            true
        } else {
            false
        }
    }

    /// Load state from disk. The state file must already exist; callers
    /// must bootstrap via [`Self::bootstrap_from_root`] on first run.
    ///
    /// # Errors
    ///
    /// Returns [`TrustedStateError`] on I/O, parse, root-verification, or
    /// key-set-mismatch failures. None of these reset `stored_timestamp_version`
    /// to `0`; they fail closed so rollback protection is preserved.
    pub fn load(path: &Path) -> Result<Self, TrustedStateError> {
        let bytes = std::fs::read(path).map_err(|e| TrustedStateError::Io {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
        let state: TrustedCatalogState =
            serde_json::from_slice(&bytes).map_err(|e| TrustedStateError::Parse {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?;

        // Re-derive keys from stored root and compare.
        let derived = super::tuf::verify_root(&state.root, None)
            .map_err(|e: CatalogError| TrustedStateError::RootVerification {
                message: e.to_string(),
            })?;
        let mut missing = Vec::new();
        for id in state.root_keys.keys() {
            if !derived.contains_key(id) {
                missing.push(id.clone());
            }
        }
        let mut unexpected = Vec::new();
        for id in derived.keys() {
            if !state.root_keys.contains_key(id) {
                unexpected.push(id.clone());
            }
        }
        if !missing.is_empty() || !unexpected.is_empty() {
            return Err(TrustedStateError::RootKeysMismatch { missing, unexpected });
        }
        Ok(state)
    }

    /// Atomically save the state to `path`. Writes to a same-directory
    /// temp file, `sync_all`, then renames over the destination.
    ///
    /// # Errors
    ///
    /// Returns [`TrustedStateError::Io`] on any filesystem failure.
    pub fn save(&self, path: &Path) -> Result<(), TrustedStateError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| TrustedStateError::Io {
                path: parent.to_path_buf(),
                message: format!("create_dir_all: {e}"),
            })?;
        }
        let bytes = serde_json::to_vec(self).map_err(|e| TrustedStateError::Io {
            path: path.to_path_buf(),
            message: format!("serialize: {e}"),
        })?;

        let tmp = path.with_extension("json.tmp");
        {
            let mut f = std::fs::File::create(&tmp).map_err(|e| TrustedStateError::Io {
                path: tmp.clone(),
                message: format!("create temp: {e}"),
            })?;
            f.write_all(&bytes).map_err(|e| TrustedStateError::Io {
                path: tmp.clone(),
                message: format!("write: {e}"),
            })?;
            f.sync_all().map_err(|e| TrustedStateError::Io {
                path: tmp.clone(),
                message: format!("sync_all: {e}"),
            })?;
        }
        std::fs::rename(&tmp, path).map_err(|e| TrustedStateError::Io {
            path: path.to_path_buf(),
            message: format!("rename: {e}"),
        })?;
        Ok(())
    }
}

/// Load [`TrustedCatalogState`] from the canonical state file.
///
/// Convenience wrapper that maps [`TrustedStateError`] to the unified
/// catalog error type so callers can use a single error path.
pub fn load_trusted_state(
    path: &Path,
) -> Result<TrustedCatalogState, CatalogError> {
    TrustedCatalogState::load(path).map_err(|e| CatalogError::State {
        path: path.to_path_buf(),
        message: e.to_string(),
    })
}

/// Save the state to the canonical state file.
pub fn save_trusted_state(
    state: &TrustedCatalogState,
    path: &Path,
) -> Result<(), CatalogError> {
    state.save(path).map_err(|e| CatalogError::State {
        path: path.to_path_buf(),
        message: e.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::builder::SigningKeyProvider;
    use crate::catalog::builder::{RoleDistinctTestKey, TestSigningKey};
    use crate::catalog::tuf::RoleConfig;

    fn make_root() -> (RootMetadata, HashMap<String, TufKey>) {
        let root_key = TestSigningKey::new();
        let targets_key = RoleDistinctTestKey::new(crate::catalog::builder::OnlineSigningRole::Targets);
        let snapshot_key = RoleDistinctTestKey::new(crate::catalog::builder::OnlineSigningRole::Snapshot);
        let timestamp_key = RoleDistinctTestKey::new(crate::catalog::builder::OnlineSigningRole::Timestamp);

        let mut keys = HashMap::new();
        keys.insert(root_key.key_id(), root_key.tuf_key());
        keys.insert(targets_key.key_id(), targets_key.tuf_key());
        keys.insert(snapshot_key.key_id(), snapshot_key.tuf_key());
        keys.insert(timestamp_key.key_id(), timestamp_key.tuf_key());

        let root_signed = crate::catalog::tuf::RootSigned {
            kind: "root".to_string(),
            version: 1,
            expires: "2999-12-31T23:59:59Z".to_string(),
            keys: keys.clone(),
            roles: HashMap::from([
                ("root".to_string(), RoleConfig {
                    keyids: vec![root_key.key_id()],
                    threshold: 1,
                }),
                ("targets".to_string(), RoleConfig {
                    keyids: vec![root_key.key_id()],
                    threshold: 1,
                }),
                ("snapshot".to_string(), RoleConfig {
                    keyids: vec![root_key.key_id()],
                    threshold: 1,
                }),
                ("timestamp".to_string(), RoleConfig {
                    keyids: vec![root_key.key_id()],
                    threshold: 1,
                }),
            ]),
        };
        // Sign the root payload with the root key
        let root_payload = serde_json::to_vec(
            &serde_json::to_value(&root_signed).unwrap()
        ).unwrap();
        let sig = root_key.sign(&root_payload);

        let root = RootMetadata {
            signed: root_signed,
            signatures: vec![crate::catalog::tuf::SignatureEntry {
                keyid: root_key.key_id(),
                sig,
            }],
        };

        let derived = super::super::tuf::verify_root(&root, None).expect("root verify");
        (root, derived)
    }

    #[test]
    fn round_trip_preserves_state() {
        let (root, keys) = make_root();
        let state = TrustedCatalogState::bootstrap_from_root(root.clone(), keys.clone());
        let mut state = state;
        state.record_timestamp_version(7);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("catalog_state.json");
        state.save(&path).expect("save");

        let restored = TrustedCatalogState::load(&path).expect("load");
        assert_eq!(restored.stored_timestamp_version, 7);
        assert_eq!(restored.root.signed.version, root.signed.version);
        assert_eq!(restored.root_keys.len(), keys.len());
    }

    #[test]
    fn corrupt_state_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("catalog_state.json");
        std::fs::write(&path, b"{ this is not valid json").unwrap();

        let result = TrustedCatalogState::load(&path);
        assert!(matches!(result, Err(TrustedStateError::Parse { .. })));
    }

    #[test]
    fn missing_state_does_not_reset_existing() {
        // Caller is responsible for bootstrapping on first run. This test
        // documents that `load` does NOT silently create or reset state.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does-not-exist.json");
        let result = TrustedCatalogState::load(&path);
        assert!(matches!(result, Err(TrustedStateError::Io { .. })));
    }

    #[test]
    fn record_timestamp_advances_only_when_strictly_greater() {
        let (root, keys) = make_root();
        let mut state = TrustedCatalogState::bootstrap_from_root(root, keys);
        assert_eq!(state.stored_timestamp_version, 0);
        assert!(state.record_timestamp_version(5));
        assert_eq!(state.stored_timestamp_version, 5);
        assert!(!state.record_timestamp_version(5));
        assert_eq!(state.stored_timestamp_version, 5);
        assert!(state.record_timestamp_version(10));
        assert_eq!(state.stored_timestamp_version, 10);
    }

    #[test]
    fn atomic_save_does_not_leave_partial_file() {
        let (root, keys) = make_root();
        let state = TrustedCatalogState::bootstrap_from_root(root, keys);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("catalog_state.json");
        state.save(&path).expect("save");
        // Temp file must be cleaned up after successful rename
        assert!(!path.with_extension("json.tmp").exists());
        assert!(path.exists());
    }
}