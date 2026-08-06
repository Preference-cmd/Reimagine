//! Bearer-token authentication for the gRPC worker transport.
//!
//! Tokens travel in the standard `authorization: Bearer <token>` metadata
//! header on every RPC (including the `Communication` bidi stream and
//! `HealthCheck`), so no proto change is required.
//!
//! Sourcing: env `REIMAGINE_WORKER_TOKEN`, or an explicit value passed by
//! the caller (e.g. from workspace config). When no token is configured
//! the transport stays plain/open for backward compatibility.

use tonic::Status;
use tonic::metadata::MetadataMap;

/// The metadata header carrying the bearer token.
pub const AUTH_HEADER: &str = "authorization";

/// The value prefix for bearer tokens.
pub const BEARER_PREFIX: &str = "Bearer ";

/// Environment variable holding the worker bearer token (cloud deployments).
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
/// `expected == None` is the backward-compat mode: the server accepts
/// any request (plain transport). `expected == Some(token)` rejects
/// missing, malformed, or mismatched tokens with `Status::unauthenticated`.
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

/// Build a tonic interceptor that attaches `authorization: Bearer <token>`
/// to every outgoing request (including streaming ones).
///
/// `None` produces an identity interceptor (plain mode).
pub fn bearer_interceptor(
    token: Option<String>,
) -> impl Fn(tonic::Request<()>) -> Result<tonic::Request<()>, Status> + Clone {
    move |mut request: tonic::Request<()>| {
        if let Some(token) = &token {
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
        let interceptor = bearer_interceptor(Some("s3cret".to_owned()));
        let request = interceptor(tonic::Request::new(())).unwrap();
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
        let interceptor = bearer_interceptor(None);
        let request = interceptor(tonic::Request::new(())).unwrap();
        assert!(request.metadata().get(AUTH_HEADER).is_none());
    }

    #[test]
    fn token_env_parsing() {
        unsafe {
            std::env::remove_var(TOKEN_ENV);
        }
        assert_eq!(token_from_env(), None);
        unsafe {
            std::env::set_var(TOKEN_ENV, "s3cret");
        }
        assert_eq!(token_from_env().as_deref(), Some("s3cret"));
        unsafe {
            std::env::set_var(TOKEN_ENV, "");
        }
        assert_eq!(token_from_env(), None);
        unsafe {
            std::env::remove_var(TOKEN_ENV);
        }
    }
}
