use std::sync::{Arc, RwLock};

use reimagine_config::AppPaths;
use reimagine_core::model::{ModelId, ModelRef};
use reimagine_model_manager::{
    ManifestModelResolver, ManifestValidationReport, ModelDescriptor, ModelDescriptorResolver,
    ModelManifest, ModelManifestStore, ModelReadinessResolver, ModelResolution, ResolvedModelInfo,
};

use crate::AppHostResult;

pub struct ModelService {
    app_paths: AppPaths,
    store: Arc<ModelManifestStore>,
    manifest: RwLock<Option<ModelManifest>>,
    report: RwLock<Option<ManifestValidationReport>>,
}

impl std::fmt::Debug for ModelService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelService")
            .field("models_dir", &self.app_paths.models_dir())
            .field("manifest_path", &self.store.path())
            .finish_non_exhaustive()
    }
}

impl ModelService {
    pub fn new(app_paths: AppPaths) -> Self {
        Self {
            store: Arc::new(ModelManifestStore::new(app_paths.clone())),
            app_paths,
            manifest: RwLock::new(None),
            report: RwLock::new(None),
        }
    }

    pub fn store(&self) -> &Arc<ModelManifestStore> {
        &self.store
    }

    pub async fn load_manifest(&self) -> AppHostResult<(ModelManifest, ManifestValidationReport)> {
        let (manifest, report) = self.store.load().await?;
        self.replace_cached_manifest(manifest.clone(), report.clone());
        Ok((manifest, report))
    }

    pub async fn save_manifest(
        &self,
        manifest: &ModelManifest,
    ) -> AppHostResult<ManifestValidationReport> {
        let report = self.store.save(manifest).await?;
        self.replace_cached_manifest(manifest.clone(), report.clone());
        Ok(report)
    }

    pub async fn list_models(&self) -> AppHostResult<Vec<ModelDescriptor>> {
        let (manifest, _) = self.load_manifest().await?;
        Ok(manifest.models().to_vec())
    }

    pub async fn remove_model(
        &self,
        model_id: &ModelId,
    ) -> AppHostResult<(ModelManifest, ManifestValidationReport)> {
        let (manifest, report) = self.store.remove_model(model_id).await?;
        self.replace_cached_manifest(manifest.clone(), report.clone());
        Ok((manifest, report))
    }

    pub async fn resolve_readiness(
        &self,
        model_ref: &ModelRef,
    ) -> AppHostResult<ModelResolution<ResolvedModelInfo>> {
        let (manifest, _) = self.load_manifest().await?;
        let resolver = ManifestModelResolver::new(&manifest, self.app_paths.models_dir());
        Ok(resolver.resolve_readiness(model_ref).await)
    }

    pub async fn resolve_descriptor(
        &self,
        model_ref: &ModelRef,
    ) -> AppHostResult<ModelResolution<ModelDescriptor>> {
        let (manifest, _) = self.load_manifest().await?;
        let resolver = ManifestModelResolver::new(&manifest, self.app_paths.models_dir());
        Ok(resolver.resolve_descriptor(model_ref).await)
    }

    pub async fn build_readiness_snapshot(
        &self,
        workflow: &reimagine_core::workflow::Workflow,
    ) -> AppHostResult<crate::readiness::SnapshotExternalReadinessProvider> {
        use reimagine_core::readiness::ExternalReadinessSubject;
        let (manifest, _) = self.load_manifest().await?;
        let models_dir = self.app_paths.models_dir().to_path_buf();
        let resolver = reimagine_model_manager::ManifestModelResolver::new(&manifest, models_dir);

        let mut provider = crate::readiness::SnapshotExternalReadinessProvider::new();
        for model_ref in collect_model_refs(workflow) {
            let subject = ExternalReadinessSubject::ModelRef(model_ref.clone());
            let resolution = resolver.resolve_readiness(&model_ref).await;
            let diagnostics = resolution.report().diagnostics().to_vec();
            match resolution.into_value() {
                Some(_info) if diagnostics.is_empty() => provider.record_ok(subject),
                Some(_) | None => {
                    provider.insert(subject, diagnostics);
                }
            }
        }
        Ok(provider)
    }

    pub fn cached_manifest(&self) -> Option<ModelManifest> {
        self.manifest
            .read()
            .expect("model manifest cache poisoned")
            .clone()
    }

    pub fn cached_report(&self) -> Option<ManifestValidationReport> {
        self.report
            .read()
            .expect("model manifest report cache poisoned")
            .clone()
    }

    fn replace_cached_manifest(&self, manifest: ModelManifest, report: ManifestValidationReport) {
        *self
            .manifest
            .write()
            .expect("model manifest cache poisoned") = Some(manifest);
        *self
            .report
            .write()
            .expect("model manifest report cache poisoned") = Some(report);
    }
}

fn collect_model_refs(workflow: &reimagine_core::workflow::Workflow) -> Vec<ModelRef> {
    use reimagine_core::model::ParamValue;
    let mut refs = Vec::new();
    for node in workflow.nodes() {
        for value in node.params().values() {
            if let ParamValue::ModelRef(model_ref) = value
                && !refs.contains(model_ref)
            {
                refs.push(model_ref.clone());
            }
        }
    }
    refs
}
