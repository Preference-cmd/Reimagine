//! Bearer-token authentication for the gRPC worker transport.
//!
//! Tokens travel in the standard `authorization: Bearer <token>` metadata
//! header on every RPC, so no proto change is required for auth. Sourcing:
//! an explicit value passed by the caller (the host generates a random
//! token per worker launch) or the `REIMAGINE_WORKER_TOKEN` environment
//! variable for remote deployments. A worker that is configured with a
//! token rejects every call that does not present exactly that token.

use tonic::Status;
use tonic::metadata::MetadataMap;
use tonic::service::Interceptor;

/// Authentication settings for a gRPC worker connection.
#[derive(Clone, Debug, Default)]
pub struct GrpcAuth {
    /// Bearer token attached as `authorization: Bearer <token>` on every
    /// request. `None` keeps the transport open (used by workers started
    /// without a token).
    pub token: Option<String>,
    /// TLS mode; `None` keeps plain HTTP. Local worker connections do
    /// not use TLS.
    pub tls: Option<crate::tls::GrpcTls>,
}

impl GrpcAuth {
    /// Open connection without a token.
    #[must_use]
    pub fn plain() -> Self {
        Self::default()
    }

    /// Read the token from the `REIMAGINE_WORKER_TOKEN` environment
    /// variable (unset/empty => no token).
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            token: token_from_env(),
            tls: None,
        }
    }
}

/// The metadata header carrying the bearer token.
pub const AUTH_HEADER: &str = "authorization";

/// The value prefix for bearer tokens.
pub const BEARER_PREFIX: &str = "Bearer ";

/// Environment variable holding the worker bearer token (remote deployments).
pub const TOKEN_ENV: &str = "REIMAGINE_WORKER_TOKEN";

/// Read the worker token from [`TOKEN_ENV`]; an unset or empty value is
/// treated as "no token configured".
#[must_use]
pub fn token_from_env() -> Option<String> {
    match std::env::var(TOKEN_ENV) {
        Ok(value) if !value.is_empty() => Some(value),
        _ => None,
    }
}

/// `len` bytes from the operating system's random number generator,
/// hex-encoded.
///
/// Used for per-launch bearer tokens and private endpoint names, which
/// must be unguessable by other local processes.
pub fn random_hex(len: usize) -> Result<String, String> {
    let mut bytes = vec![0_u8; len];
    rustls::crypto::ring::default_provider()
        .secure_random
        .fill(&mut bytes)
        .map_err(|_| "the operating system random number generator failed".to_owned())?;
    Ok(hex::encode(bytes))
}

/// Constant-time equality over two strings.
#[must_use]
pub fn ct_eq(a: &str, b: &str) -> bool {
    let a = a.as_bytes();
    let b = b.as_bytes();
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Validate the `authorization: Bearer <token>` metadata against
/// `expected`.
///
/// `expected == None` accepts any request (token-auth disabled). With an
/// expected token, missing, malformed, or mismatched tokens are rejected
/// with `Status::unauthenticated`.
pub fn check_bearer(metadata: &MetadataMap, expected: Option<&str>) -> Result<(), Status> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let presented = metadata
        .get(AUTH_HEADER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix(BEARER_PREFIX));
    match presented {
        Some(token) if ct_eq(token, expected) => Ok(()),
        _ => Err(Status::unauthenticated("missing or invalid bearer token")),
    }
}

/// Interceptor that attaches `authorization: Bearer <token>` to every
/// outgoing request.
///
/// `None` produces an identity interceptor (no token configured).
#[derive(Clone, Debug, Default)]
pub struct BearerInterceptor {
    token: Option<String>,
}

impl BearerInterceptor {
    #[must_use]
    pub fn new(token: Option<String>) -> Self {
        Self { token }
    }
}

impl Interceptor for BearerInterceptor {
    fn call(&mut self, mut request: tonic::Request<()>) -> Result<tonic::Request<()>, Status> {
        if let Some(token) = &self.token {
            let value = format!("{BEARER_PREFIX}{token}")
                .parse::<tonic::metadata::MetadataValue<_>>()
                .map_err(|_| Status::internal("worker token is not valid header material"))?;
            request.metadata_mut().insert(AUTH_HEADER, value);
        }
        Ok(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tonic::metadata::MetadataMap;

    #[test]
    fn random_hex_is_unique_and_sized() {
        let first = random_hex(32).unwrap();
        assert_eq!(first.len(), 64);
        assert_ne!(first, random_hex(32).unwrap());
    }

    #[test]
    fn ct_eq_compares_constantly() {
        assert!(ct_eq("abc", "abc"));
        assert!(!ct_eq("abc", "abd"));
        assert!(!ct_eq("abc", "abcd"));
        assert!(!ct_eq("abc", ""));
    }

    #[test]
    fn open_mode_accepts_anything() {
        let metadata = MetadataMap::new();
        assert!(check_bearer(&metadata, None).is_ok());
    }

    #[test]
    fn missing_or_wrong_token_rejected() {
        let metadata = MetadataMap::new();
        assert_eq!(
            check_bearer(&metadata, Some("s3cret")).unwrap_err().code(),
            tonic::Code::Unauthenticated
        );

        let mut metadata = MetadataMap::new();
        metadata.insert(AUTH_HEADER, "Bearer wrong".parse().unwrap());
        assert_eq!(
            check_bearer(&metadata, Some("s3cret")).unwrap_err().code(),
            tonic::Code::Unauthenticated
        );
    }

    #[test]
    fn correct_token_accepted() {
        let mut metadata = MetadataMap::new();
        metadata.insert(AUTH_HEADER, "Bearer s3cret".parse().unwrap());
        assert!(check_bearer(&metadata, Some("s3cret")).is_ok());
    }

    #[test]
    fn interceptor_attaches_token() {
        let mut interceptor = BearerInterceptor::new(Some("s3cret".to_owned()));
        let request = interceptor.call(tonic::Request::new(())).unwrap();
        let value = request
            .metadata()
            .get(AUTH_HEADER)
            .unwrap()
            .to_str()
            .unwrap();
        assert_eq!(value, "Bearer s3cret");
    }

    #[test]
    fn interceptor_without_token_is_identity() {
        let mut interceptor = BearerInterceptor::new(None);
        let request = interceptor.call(tonic::Request::new(())).unwrap();
        assert!(request.metadata().get(AUTH_HEADER).is_none());
    }
}
