//! Node catalog and executor alignment service.
//!
//! This module owns the host-facing surface for the built-in `NodeDef`
//! catalog and the alignment check between the catalog and the runtime
//! executor registry. The catalog is the single source of truth for node
//! metadata (slots, params, outputs, effect) and the executor registry is
//! the single source of truth for execution behavior. The two are
//! assembled in different crates (`reimagine-nodes` and
//! `reimagine-inference`) and composed in `app-host`; this service
//! makes that composition observable.
//!
//! ## Required shape
//!
//! - `core` owns the `NodeDef` schema types.
//! - `reimagine-nodes` owns the V1 built-in catalog data.
//! - `app-host` exposes the workspace catalog handle and the alignment
//!   report. UI, Tauri, Axum, and Agent tools must read node metadata
//!   from the catalog exposed here, never from a parallel hard-coded
//!   list.
//! - The runtime executor registry remains execution behavior only. The
//!   registry is *not* a node metadata source.
//!
//! See `.scratch/app-host/issues/05-node-catalog-service-and-executor-alignment.md`
//! and `docs/architecture/modules/app-host.md` for the spec.

use std::sync::Arc;

use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::model::{DiagnosticId, NodeCatalog, NodeDef, NodeTypeId};
use reimagine_inference::NodeExecutorRegistry;
use reimagine_nodes::BuiltinNodeCatalog;

use crate::BackendSelection;

/// Host-facing service for the built-in node catalog.
///
/// Wraps the workspace [`BuiltinNodeCatalog`] (V1's only catalog
/// surface) together with the selected [`BackendSelection`] so host
/// adapters can list and fetch `NodeDef` entries through a single
/// host-neutral API and run catalog/executor alignment checks.
///
/// V1 keeps the catalog type concrete. A future V2 catalog surface
/// (e.g. workspace-loaded plugins) will replace this struct.
#[derive(Debug, Clone)]
pub struct NodeCatalogService {
    catalog: Arc<BuiltinNodeCatalog>,
    backend: BackendSelection,
}

impl NodeCatalogService {
    /// Build a service over the given built-in catalog and selected
    /// backend profile.
    pub fn new(catalog: Arc<BuiltinNodeCatalog>, backend: BackendSelection) -> Self {
        Self { catalog, backend }
    }

    /// List every `NodeDef` exposed by the workspace catalog.
    ///
    /// The returned entries are clones; callers (UI, Tauri, Axum, Agent
    /// tools, import adapters) must not mutate them and must not derive
    /// node metadata from any other source.
    pub fn list_node_defs(&self) -> Vec<NodeDef> {
        self.catalog.iter().cloned().collect()
    }

    /// Fetch a single `NodeDef` by `NodeTypeId`.
    pub fn find_node_def(&self, type_id: &NodeTypeId) -> Option<NodeDef> {
        self.catalog.get(type_id).cloned()
    }

    /// Borrow the selected backend profile used to label diagnostics.
    pub fn backend(&self) -> BackendSelection {
        self.backend
    }

    /// Borrow the underlying catalog handle.
    pub fn builtin_catalog(&self) -> &Arc<BuiltinNodeCatalog> {
        &self.catalog
    }

    /// Compute the alignment report between the catalog and the runtime
    /// executor registry.
    ///
    /// Two kinds of mismatch are reported:
    /// - `missing_executors`: catalog entries that have no registered
    ///   executor for the selected backend. Workflows that reach these
    ///   nodes will fail at run time.
    /// - `orphan_executors`: registered executors that have no catalog
    ///   entry. Such executors can never be referenced by a valid
    ///   workflow and usually indicate that the catalog and registry
    ///   drifted out of sync.
    pub fn check_alignment(
        &self,
        executor_registry: &NodeExecutorRegistry,
    ) -> NodeCatalogAlignment {
        let mut missing_executors: Vec<NodeTypeId> = self
            .catalog
            .iter()
            .map(|def| def.type_id().clone())
            .filter(|id| executor_registry.get(id).is_none())
            .collect();
        missing_executors.sort();

        let mut orphan_executors: Vec<NodeTypeId> = executor_registry
            .iter_type_ids()
            .filter(|id| self.catalog.get(id).is_none())
            .cloned()
            .collect();
        orphan_executors.sort();

        NodeCatalogAlignment {
            backend: self.backend,
            missing_executors,
            orphan_executors,
        }
    }
}

impl NodeCatalog for NodeCatalogService {
    fn get(&self, type_id: &NodeTypeId) -> Option<&NodeDef> {
        self.catalog.get(type_id)
    }
}

/// Result of comparing the catalog against the runtime executor registry.
///
/// Construct via [`NodeCatalogService::check_alignment`]. The fields are
/// sorted to keep diagnostics stable across runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeCatalogAlignment {
    backend: BackendSelection,
    missing_executors: Vec<NodeTypeId>,
    orphan_executors: Vec<NodeTypeId>,
}

impl NodeCatalogAlignment {
    /// Returns `true` when every catalog entry has a matching executor
    /// and every executor has a matching catalog entry.
    pub fn is_aligned(&self) -> bool {
        self.missing_executors.is_empty() && self.orphan_executors.is_empty()
    }

    /// Borrow the backend profile the alignment was computed against.
    pub fn backend(&self) -> BackendSelection {
        self.backend
    }

    /// Catalog entries that have no registered executor for the
    /// selected backend.
    pub fn missing_executors(&self) -> &[NodeTypeId] {
        &self.missing_executors
    }

    /// Registered executors that have no catalog entry.
    pub fn orphan_executors(&self) -> &[NodeTypeId] {
        &self.orphan_executors
    }

    /// Borrow the diagnostics as the host-facing `Diagnostic` stream.
    ///
    /// The diagnostics reference the missing/orphan `NodeTypeId` and
    /// the selected backend profile. They do not leak backend-internal
    /// types or kernel details.
    pub fn diagnostics(&self) -> Vec<Diagnostic> {
        let backend_label = self.backend.to_string();
        let mut out =
            Vec::with_capacity(self.missing_executors.len() + self.orphan_executors.len());
        for type_id in &self.missing_executors {
            out.push(missing_executor_diagnostic(type_id.clone(), &backend_label));
        }
        for type_id in &self.orphan_executors {
            out.push(orphan_executor_diagnostic(type_id.clone(), &backend_label));
        }
        out
    }

    /// Consume the report and return the diagnostics stream.
    pub fn into_diagnostics(self) -> Vec<Diagnostic> {
        self.diagnostics()
    }
}

/// Build a diagnostic for a catalog entry that has no registered
/// executor for the selected backend profile.
fn missing_executor_diagnostic(type_id: NodeTypeId, backend: &str) -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new(format!(
            "app-host:catalog:missing_executor:{}",
            type_id.as_str()
        )),
        DiagnosticCode::new("APP_HOST/NODE_CATALOG_MISSING_EXECUTOR"),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("app-host"),
        format!(
            "node type `{}` is in the workspace catalog but has no executor registered for backend `{}`",
            type_id.as_str(),
            backend,
        ),
        target_for_node_type(&type_id),
    )
}

/// Build a diagnostic for a registered executor that has no catalog
/// entry.
fn orphan_executor_diagnostic(type_id: NodeTypeId, backend: &str) -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new(format!(
            "app-host:catalog:orphan_executor:{}",
            type_id.as_str()
        )),
        DiagnosticCode::new("APP_HOST/NODE_CATALOG_ORPHAN_EXECUTOR"),
        DiagnosticSeverity::Warning,
        DiagnosticSourceName::new("app-host"),
        format!(
            "executor for `{}` is registered for backend `{}` but has no catalog entry",
            type_id.as_str(),
            backend,
        ),
        target_for_node_type(&type_id),
    )
}

fn target_for_node_type(type_id: &NodeTypeId) -> DiagnosticTarget {
    DiagnosticTarget::new(DiagnosticTargetDomain::new("app-host.node_catalog"))
        .with_id(type_id.as_str().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_core::model::{InputSlotDef, OutputSlotDef, SlotKind};
    use reimagine_nodes::{
        BUILTIN_CHECKPOINT_LOADER, BUILTIN_KSAMPLER, BUILTIN_STRING, BuiltinNodeCatalog,
    };

    fn test_catalog() -> Arc<BuiltinNodeCatalog> {
        Arc::new(BuiltinNodeCatalog::v1())
    }

    fn builtin_string() -> NodeDef {
        NodeDef::new(BUILTIN_STRING, "String", "input")
            .with_input_slot(InputSlotDef::new("value", SlotKind::String))
            .with_output_slot(OutputSlotDef::new("value", SlotKind::String))
    }

    fn builtin_checkpoint() -> NodeDef {
        NodeDef::new(BUILTIN_CHECKPOINT_LOADER, "Checkpoint Loader", "model")
    }

    #[test]
    fn empty_registry_marks_all_catalog_entries_as_missing() {
        let catalog = test_catalog();
        let service = NodeCatalogService::new(catalog.clone(), BackendSelection::Candle);
        let registry = NodeExecutorRegistry::default();

        let report = service.check_alignment(&registry);
        assert!(!report.is_aligned());
        assert_eq!(report.missing_executors().len(), catalog.len());
        assert!(report.orphan_executors().is_empty());

        let diagnostics = report.into_diagnostics();
        assert_eq!(diagnostics.len(), catalog.len());
        assert!(
            diagnostics
                .iter()
                .all(|d| d.code().as_str() == "APP_HOST/NODE_CATALOG_MISSING_EXECUTOR")
        );
        assert!(
            diagnostics
                .iter()
                .all(|d| d.severity() == DiagnosticSeverity::Error)
        );
    }

    #[test]
    fn fully_aligned_catalog_and_registry_report_no_diagnostics() {
        let catalog = test_catalog();
        let service = NodeCatalogService::new(catalog.clone(), BackendSelection::Candle);

        // Build a registry that mirrors every entry in the catalog.
        let mut registry = NodeExecutorRegistry::default();
        for def in catalog.iter() {
            registry
                .register(def.type_id().clone(), Arc::new(NoopExecutor))
                .expect("register");
        }

        let report = service.check_alignment(&registry);
        assert!(report.is_aligned());
        assert!(report.missing_executors().is_empty());
        assert!(report.orphan_executors().is_empty());
        assert!(report.into_diagnostics().is_empty());
    }

    #[test]
    fn orphan_executors_are_reported_as_warnings() {
        // A catalog that only exposes `builtin.string`.
        let catalog = Arc::new(BuiltinNodeCatalog::new(vec![builtin_string()]));
        let service = NodeCatalogService::new(catalog.clone(), BackendSelection::Candle);

        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(NodeTypeId::new(BUILTIN_STRING), Arc::new(NoopExecutor))
            .expect("register string");
        registry
            .register(
                NodeTypeId::new(BUILTIN_CHECKPOINT_LOADER),
                Arc::new(NoopExecutor),
            )
            .expect("register orphan");

        let report = service.check_alignment(&registry);
        assert!(!report.is_aligned());
        assert!(report.missing_executors().is_empty());
        assert_eq!(
            report.orphan_executors(),
            &[NodeTypeId::new(BUILTIN_CHECKPOINT_LOADER)]
        );

        let diagnostics = report.diagnostics();
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].code().as_str(),
            "APP_HOST/NODE_CATALOG_ORPHAN_EXECUTOR"
        );
        assert_eq!(diagnostics[0].severity(), DiagnosticSeverity::Warning);
    }

    #[test]
    fn alignment_reports_both_missing_and_orphan_when_drifted() {
        let catalog = Arc::new(BuiltinNodeCatalog::new(vec![
            builtin_string(),
            builtin_checkpoint(),
        ]));
        let service = NodeCatalogService::new(catalog, BackendSelection::Candle);

        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(NodeTypeId::new(BUILTIN_STRING), Arc::new(NoopExecutor))
            .expect("register string");
        // ksampler is registered, but is not in this trimmed catalog → orphan.
        registry
            .register(NodeTypeId::new(BUILTIN_KSAMPLER), Arc::new(NoopExecutor))
            .expect("register orphan ksampler");

        let report = service.check_alignment(&registry);
        assert!(!report.is_aligned());
        assert_eq!(
            report.missing_executors(),
            &[NodeTypeId::new(BUILTIN_CHECKPOINT_LOADER)]
        );
        assert_eq!(
            report.orphan_executors(),
            &[NodeTypeId::new(BUILTIN_KSAMPLER)]
        );
    }

    #[test]
    fn list_node_defs_and_find_node_def_return_clones() {
        let catalog = test_catalog();
        let service = NodeCatalogService::new(catalog.clone(), BackendSelection::Candle);

        let defs = service.list_node_defs();
        assert_eq!(defs.len(), catalog.len());

        let found = service
            .find_node_def(&NodeTypeId::new(BUILTIN_STRING))
            .expect("string def should be present");
        assert_eq!(found.type_id().as_str(), BUILTIN_STRING);

        assert!(
            service
                .find_node_def(&NodeTypeId::new("builtin.does_not_exist"))
                .is_none()
        );
    }

    struct NoopExecutor;

    #[async_trait::async_trait]
    impl reimagine_inference::NodeExecutor for NoopExecutor {
        async fn execute(
            &self,
            _context: reimagine_inference::NodeExecutionContext,
        ) -> Result<reimagine_inference::NodeExecutionOutputs, reimagine_inference::NodeExecutorError>
        {
            Ok(Vec::new())
        }
    }
}
