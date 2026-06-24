//! Runnable Axum development server for the Reimagine peer host adapter.
//!
//! The binary is intentionally thin: CLI parsing, tracing setup, and
//! workspace bootstrap live in the `reimagine-axum-host` library so they
//! can be tested directly. This file only wires the pieces together and
//! blocks until a shutdown signal is received.

use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;
use reimagine_axum_host::{
    AxumHostState, Cli, RunEventRecorder, bootstrap_workspace, default_workspace_path,
    ensure_workspace_dirs, init_tracing, run_server,
};

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    init_tracing(cli.log_filter.as_deref());

    let base_path = cli.base_path.unwrap_or_else(|| {
        let path = default_workspace_path();
        eprintln!(
            "No --base-path provided; using executable-adjacent workspace: {}",
            path.display()
        );
        path
    });

    if let Err(error) = ensure_workspace_dirs(&base_path).await {
        eprintln!("failed to prepare workspace directories: {error}");
        return ExitCode::from(1);
    }

    let recorder = Arc::new(RunEventRecorder::new());
    let workspace = match bootstrap_workspace(&base_path, recorder.clone()).await {
        Ok(workspace) => workspace,
        Err(error) => {
            eprintln!("workspace bootstrap failed: {error}");
            return ExitCode::from(1);
        }
    };

    let paths = workspace.config().paths();
    tracing::info!(
        addr = %cli.addr,
        base_path = %workspace.base_path().display(),
        models_dir = %paths.models_dir().display(),
        output_dir = %paths.output_dir().display(),
        workflows_dir = %paths.workflows_dir().display(),
        config_dir = %paths.config_dir().display(),
        "Reimagine Axum host starting",
    );

    let backend_config = workspace.backend_config();
    tracing::info!(
        backend = ?backend_config.backend,
        device = %backend_config.candle_device,
        "inference backend selected",
    );

    match run_server(AxumHostState::new(workspace, recorder), cli.addr).await {
        Ok(handle) => {
            let local_addr = handle.local_addr();
            tracing::info!(addr = %local_addr, "server listening");

            if let Err(error) = tokio::signal::ctrl_c().await {
                tracing::error!(error = %error, "failed to listen for shutdown signal");
                return ExitCode::from(1);
            }

            tracing::info!("shutdown signal received");
            if let Err(error) = handle.shutdown().await {
                tracing::error!(error = %error, "server shutdown failed");
                return ExitCode::from(1);
            }

            tracing::info!("server stopped");
            ExitCode::SUCCESS
        }
        Err(error) => {
            tracing::error!(error = %error, "failed to start server");
            ExitCode::from(1)
        }
    }
}
