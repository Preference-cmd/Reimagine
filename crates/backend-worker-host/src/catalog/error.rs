use std::fmt;
use std::path::PathBuf;

#[derive(Debug)]
pub enum CatalogError {
    /// The embedded root metadata could not be loaded or parsed.
    RootLoad {
        message: String,
    },
    /// Signature verification for a role failed.
    Signature {
        role: String,
        key_id: String,
        message: String,
    },
    /// A role's metadata has expired.
    Expired {
        role: String,
        expires: String,
    },
    /// A freeze-attack check failed (timestamp made no progress).
    FreezeAttack {
        role: String,
    },
    /// A version rollback was detected.
    Rollback {
        role: String,
        stored: u64,
        attempted: u64,
    },
    /// Target hash mismatch after download.
    TargetHashMismatch {
        target: String,
        algorithm: String,
    },
    /// Target length mismatch after download.
    TargetLengthMismatch {
        target: String,
        expected: u64,
        actual: u64,
    },
    MetadataVersionMismatch {
        role: String,
        expected: u64,
        actual: u64,
    },
    MetadataLengthMismatch {
        role: String,
        expected: u64,
        actual: u64,
    },
    MetadataHashMismatch {
        role: String,
        algorithm: String,
    },
    /// Network error fetching metadata or target.
    Network {
        url: String,
        message: String,
    },
    /// JSON parse/schema error in metadata.
    Json {
        path: Option<PathBuf>,
        message: String,
    },
    /// Threshold not met for a role.
    ThresholdNotMet {
        role: String,
        required: usize,
        received: usize,
    },
    /// Unknown key referenced in metadata.
    UnknownKey {
        role: String,
        key_id: String,
    },
}

impl fmt::Display for CatalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RootLoad { message } => write!(f, "root metadata load failed: {message}"),
            Self::Signature {
                role,
                key_id,
                message,
            } => {
                write!(f, "{role} signature from key `{key_id}` failed: {message}")
            }
            Self::Expired { role, expires } => {
                write!(f, "{role} metadata expired at {expires}")
            }
            Self::FreezeAttack { role } => write!(f, "{role} freeze-attack detected"),
            Self::Rollback {
                role,
                stored,
                attempted,
            } => {
                write!(
                    f,
                    "{role} rollback detected: stored version {stored}, attempted {attempted}"
                )
            }
            Self::TargetHashMismatch { target, algorithm } => {
                write!(f, "{algorithm} hash mismatch for target `{target}`")
            }
            Self::TargetLengthMismatch {
                target,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "target `{target}` length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::MetadataVersionMismatch {
                role,
                expected,
                actual,
            } => write!(
                f,
                "{role} metadata version mismatch: expected {expected}, got {actual}"
            ),
            Self::MetadataLengthMismatch {
                role,
                expected,
                actual,
            } => write!(
                f,
                "{role} metadata length mismatch: expected {expected}, got {actual}"
            ),
            Self::MetadataHashMismatch { role, algorithm } => {
                write!(f, "{role} metadata {algorithm} hash mismatch")
            }
            Self::Network { url, message } => {
                write!(f, "network error fetching `{url}`: {message}")
            }
            Self::Json {
                path: Some(p),
                message,
            } => {
                write!(f, "JSON error at `{}`: {message}", p.display())
            }
            Self::Json {
                path: None,
                message,
            } => write!(f, "JSON error: {message}"),
            Self::ThresholdNotMet {
                role,
                required,
                received,
            } => {
                write!(
                    f,
                    "{role} threshold not met: need {required} signatures, got {received}"
                )
            }
            Self::UnknownKey { role, key_id } => {
                write!(f, "{role} references unknown key `{key_id}`")
            }
        }
    }
}

impl std::error::Error for CatalogError {}

pub type CatalogResult<T> = Result<T, CatalogError>;
