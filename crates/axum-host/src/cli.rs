//! CLI parsing and tracing initialization for the Axum host binary.
//!
//! These helpers are intentionally small and binary-oriented; they live
//! in the library so they can be unit-tested without spawning a process.

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;
use tracing_subscriber::EnvFilter;

/// Command-line arguments for the `reimagine-axum-host` dev server.
#[derive(Debug, Clone, Parser)]
#[command(name = "reimagine-axum-host")]
#[command(about = "Reimagine peer host adapter over HTTP")]
pub struct Cli {
    /// Workspace base path. If omitted, `workspace` next to the running
    /// executable is used and printed at startup.
    #[arg(long, value_name = "PATH")]
    pub base_path: Option<PathBuf>,

    /// Address to bind the HTTP listener.
    #[arg(long, value_name = "ADDR", default_value = "127.0.0.1:7878")]
    pub addr: SocketAddr,

    /// Tracing filter directive (e.g. `info,reimagine_axum_host=debug`).
    /// Overrides `RUST_LOG` when present.
    #[arg(long, value_name = "FILTER")]
    pub log_filter: Option<String>,
}

/// Build a `tracing` [`EnvFilter`] honoring `--log-filter` and `RUST_LOG`.
pub fn build_env_filter(log_filter: Option<&str>) -> EnvFilter {
    let filter = match log_filter {
        Some(value) => value.to_string(),
        None => std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
    };
    EnvFilter::try_new(&filter).unwrap_or_else(|err| {
        eprintln!("invalid log filter `{filter}`: {err}; falling back to info");
        EnvFilter::new("info")
    })
}

/// Install a shared `tracing_subscriber` formatter.
///
/// This may only be called once per process; the binary calls it from
/// `main`, and tests should avoid calling it.
pub fn init_tracing(log_filter: Option<&str>) {
    let env_filter = build_env_filter(log_filter);
    tracing_subscriber::fmt().with_env_filter(env_filter).init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_defaults() {
        let cli = Cli::parse_from(["reimagine-axum-host"]);
        assert!(cli.base_path.is_none());
        assert_eq!(cli.addr.to_string(), "127.0.0.1:7878");
        assert!(cli.log_filter.is_none());
    }

    #[test]
    fn parse_custom_args() {
        let cli = Cli::parse_from([
            "reimagine-axum-host",
            "--base-path",
            "/tmp/ws",
            "--addr",
            "0.0.0.0:9999",
            "--log-filter",
            "debug",
        ]);
        assert_eq!(cli.base_path, Some(PathBuf::from("/tmp/ws")));
        assert_eq!(cli.addr.to_string(), "0.0.0.0:9999");
        assert_eq!(cli.log_filter, Some("debug".to_string()));
    }

    #[test]
    fn build_env_filter_uses_explicit_value() {
        let filter = build_env_filter(Some("reimagine_axum_host=debug"));
        // `EnvFilter` parses the directive without panicking.
        let _ = filter;
    }

    #[test]
    fn build_env_filter_falls_back_to_info_on_invalid_value() {
        let filter = build_env_filter(Some("not a valid filter!!!"));
        // The fallback filter is usable.
        let _ = filter;
    }
}
