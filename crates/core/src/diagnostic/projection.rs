use super::{Diagnostic, DiagnosticRelated, DiagnosticTarget};

pub fn project_diagnostic(original: &Diagnostic, new_primary: DiagnosticTarget) -> Diagnostic {
    let mut projected = Diagnostic::new(
        original.id().clone(),
        original.code().clone(),
        original.severity(),
        original.source().clone(),
        original.message().to_owned(),
        new_primary,
    );

    if let Some(correlation_id) = original.correlation_id() {
        projected = projected.with_correlation_id(correlation_id.clone());
    }

    if let Some(trace_span_id) = original.trace_span_id() {
        projected = projected.with_trace_span_id(trace_span_id.to_owned());
    }

    projected = projected.with_related(DiagnosticRelated::new(
        original.primary().clone(),
        "external readiness source",
    ));

    for related in original.related() {
        projected = projected.with_related(related.clone());
    }

    for fix in original.fixes() {
        projected = projected.with_fix(fix.clone());
    }

    projected
}
