use reimagine_core::diagnostic::{
    CorrelationId, Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName,
    DiagnosticTarget, DiagnosticTargetDomain, project_diagnostic,
};
use reimagine_core::event::{RunEvent, RunEventKind, Timestamp};
use reimagine_core::model::{ArtifactId, DiagnosticId, RunId, WorkflowId, WorkflowVersion};

#[test]
fn projected_diagnostic_retargets_workflow_context_and_preserves_external_cause() {
    let original = Diagnostic::new(
        DiagnosticId::new("diag-model-stale"),
        DiagnosticCode::new("MODEL_MANAGER/MODEL_SOURCE_STALE"),
        DiagnosticSeverity::Warning,
        DiagnosticSourceName::new("model-manager"),
        "model source is stale",
        DiagnosticTarget::new(DiagnosticTargetDomain::new("model")).with_id("sdxl-base-1.0"),
    )
    .with_correlation_id(CorrelationId::new("corr-1"))
    .with_trace_span_id("span-42");

    let projected = project_diagnostic(
        &original,
        DiagnosticTarget::new(DiagnosticTargetDomain::new("workflow.node"))
            .with_id("node_checkpoint")
            .with_path("params.checkpoint"),
    );

    assert_eq!(
        projected.code().as_str(),
        "MODEL_MANAGER/MODEL_SOURCE_STALE"
    );
    assert_eq!(projected.source().as_str(), "model-manager");
    assert_eq!(projected.severity(), DiagnosticSeverity::Warning);
    assert_eq!(projected.message(), "model source is stale");
    assert_eq!(
        projected.correlation_id().map(|id| id.as_str()),
        Some("corr-1")
    );
    assert_eq!(projected.trace_span_id(), Some("span-42"));
    assert_eq!(projected.primary().domain().as_str(), "workflow.node");
    assert_eq!(projected.primary().id(), Some("node_checkpoint"));
    assert_eq!(projected.primary().path(), Some("params.checkpoint"));
    assert_eq!(projected.related().len(), 1);
    assert_eq!(projected.related()[0].target().domain().as_str(), "model");
    assert_eq!(projected.related()[0].target().id(), Some("sdxl-base-1.0"));
    assert_eq!(
        projected.related()[0].message(),
        "external readiness source"
    );
}

#[test]
fn run_event_builder_supports_v1_payload_shape() {
    let event = RunEvent::new(
        "run-event-1",
        RunId::new("run-1"),
        WorkflowId::new("workflow-1"),
        WorkflowVersion::new(7),
        RunEventKind::ArtifactCreated,
        Timestamp::new("2026-06-09T18:00:00Z"),
    )
    .with_node_id("node_save")
    .with_artifact(ArtifactId::new("artifact-1"))
    .with_correlation_id(CorrelationId::new("corr-run"))
    .with_diagnostic(Diagnostic::new(
        DiagnosticId::new("diag-run"),
        DiagnosticCode::new("RUNTIME/ARTIFACT_WRITTEN"),
        DiagnosticSeverity::Info,
        DiagnosticSourceName::new("runtime"),
        "artifact emitted",
        DiagnosticTarget::new(DiagnosticTargetDomain::new("workflow.node")).with_id("node_save"),
    ));

    assert_eq!(event.kind(), RunEventKind::ArtifactCreated);
    assert_eq!(event.node_id().map(|id| id.as_str()), Some("node_save"));
    assert_eq!(event.artifact().map(|id| id.as_str()), Some("artifact-1"));
    assert_eq!(
        event.correlation_id().map(|id| id.as_str()),
        Some("corr-run")
    );
    assert_eq!(event.diagnostics().len(), 1);
}
