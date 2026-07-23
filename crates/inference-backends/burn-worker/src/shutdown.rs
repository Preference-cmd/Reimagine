//! Deterministic cleanup and graceful exit for the Burn worker.
//!
//! On shutdown the worker drains any pending state and flushes
//! pending protocol acks before the process exits. After
//! `execute` returns the process must exit deterministically
//! (no background threads, no lingering GPU resources).

use reimagine_backend_worker_protocol::WorkerIncarnationId;
use reimagine_inference_burn::BurnBackend;

/// Perform run-scoped cleanup for the given incarnation.
///
/// V1: releases all stored payloads and cached model bundles
/// associated with this worker incarnation. After this call the
/// process is ready to terminate.
pub fn cleanup(backend: &BurnBackend, _incarnation: &WorkerIncarnationId) {
    let payloads = backend.store().cleanup_all();
    let bundles = backend.model_cache().clear();
    eprintln!(
        "worker shutdown: {} payloads released, {} cached bundles",
        payloads, bundles
    );
}
