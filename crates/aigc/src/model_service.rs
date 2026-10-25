use std::sync::{Arc, RwLock};

use reimagine_config::AppPaths;
use reimagine_core::diagnostic::Diagnostic;
use reimagine_core::model::{ModelId, ModelRef, ModelRole, ModelSeries, ModelVariant};

use reimagine_model_acquisition::{
    AcquireProvider, AllowPatterns, ComponentRole, ModelAcquisitionRequest, ModelCard,
    ModelCatalog, ModelSearchQuery, OverwritePolicy, RepoId, Revision, TargetRelativeDir,
    resolve_from_model_index,
};
use reimagine_model_manager::{
    ManifestModelResolver, ModelComponentSource, ModelDescriptor, ModelDescriptorResolver,
    ModelFormat, ModelManifest, ModelManifestStore, ModelReadinessResolver, ModelResolution,
    ModelRootId, ModelSource, ModelSourceStatus, ResolvedDescriptorView, ResolvedModelInfo,
    resolve_source_path, upsert_burn_package_descriptor,
};

use crate::AigcResult;
use crate::model_acquisition_service::ModelAcquisitionService;
use crate::{BurnCheckpointConverter, BurnConversionReport};

pub struct ModelService {
    app_paths: AppPaths,
    store: Arc<ModelManifestStore>,
    manifest: RwLock<Option<ModelManifest>>,
    diagnostics: RwLock<Option<Vec<Diagnostic>>>,
    acquisition_service: Arc<ModelAcquisitionService>,
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
    pub fn new(app_paths: AppPaths, acquisition_service: Arc<ModelAcquisitionService>) -> Self {
        Self {
            store: Arc::new(ModelManifestStore::new(app_paths.clone())),
            app_paths,
            manifest: RwLock::new(None),
            diagnostics: RwLock::new(None),
            acquisition_service,
        }
    }

    pub fn store(&self) -> &Arc<ModelManifestStore> {
        &self.store
    }

    /// Return the path to the workspace models directory.
    pub fn models_dir(&self) -> &std::path::Path {
        self.app_paths.models_dir()
    }

    pub async fn load_manifest(&self) -> AigcResult<(ModelManifest, Vec<Diagnostic>)> {
        let (manifest, report) = self.store.load().await?;
        self.replace_cached_manifest(manifest.clone(), report.clone());
        Ok((manifest, report))
    }

    pub async fn save_manifest(&self, manifest: &ModelManifest) -> AigcResult<Vec<Diagnostic>> {
        let report = self.store.save(manifest).await?;
        self.replace_cached_manifest(manifest.clone(), report.clone());
        Ok(report)
    }

    pub async fn list_models(&self) -> AigcResult<Vec<ModelDescriptor>> {
        let (manifest, _) = self.load_manifest().await?;
        Ok(manifest.models().to_vec())
    }

    pub async fn remove_model(
        &self,
        model_id: &ModelId,
    ) -> AigcResult<(ModelManifest, Vec<Diagnostic>)> {
        let (manifest, report) = self.store.remove_model(model_id).await?;
        self.replace_cached_manifest(manifest.clone(), report.clone());
        Ok((manifest, report))
    }

    pub async fn import_burn_converted_package(
        &self,
        report_path: impl AsRef<std::path::Path>,
    ) -> AigcResult<(ModelManifest, Vec<Diagnostic>, ModelDescriptor)> {
        let report_path = self.normalize_burn_package_report_path(report_path.as_ref())?;
        let (mut manifest, _) = self.load_manifest().await?;
        let descriptor = upsert_burn_package_descriptor(
            &mut manifest,
            &report_path,
            self.app_paths.models_dir(),
        )
        .await?;
        let report = self.save_manifest(&manifest).await?;
        Ok((manifest, report, descriptor))
    }

    /// Convert an already-downloaded checkpoint into Burn-native components.
    ///
    /// Downloads via ModelAcquisitionService have already been handled.
    /// This method looks up the manifest entry for the checkpoint and
    /// calls the Burn import pipeline.
    pub async fn convert_checkpoint_to_burn(
        &self,
        model_id: &str,
        converter: &dyn BurnCheckpointConverter,
    ) -> AigcResult<BurnConversionReport> {
        let (manifest, _) = self.load_manifest().await?;
        let model_id_core = reimagine_core::model::ModelId::new(model_id);
        let descriptor = manifest
            .models()
            .iter()
            .find(|d| d.id() == &model_id_core)
            .cloned()
            .ok_or_else(|| crate::AigcError::Io {
                path: self.store.path().to_path_buf(),
                message: format!("model `{model_id}` not found in manifest"),
            })?;

        let source_path =
            resolve_source_path(&manifest, descriptor.source(), self.app_paths.models_dir())
                .ok_or_else(|| crate::AigcError::Io {
                    path: self.store.path().to_path_buf(),
                    message: format!("model `{model_id}` source path cannot be resolved"),
                })?;

        let model_root = self.app_paths.models_dir();
        let report = self.convert_with_burn_port(&source_path, model_id, model_root, converter)?;
        Ok(report)
    }

    /// Convert a safetensors file path directly into Burn-native components,
    /// without requiring a manifest entry. Used by the `POST /models/acquire`
    /// flow after downloading.
    pub fn convert_safetensors_to_burn(
        &self,
        source_path: &std::path::Path,
        model_id: &str,
        converter: &dyn BurnCheckpointConverter,
    ) -> Result<BurnConversionReport, crate::AigcError> {
        let model_root = self.app_paths.models_dir();
        let report = self.convert_with_burn_port(source_path, model_id, model_root, converter)?;
        Ok(report)
    }

    fn convert_with_burn_port(
        &self,
        source_path: &std::path::Path,
        model_id: &str,
        model_root: &std::path::Path,
        converter: &dyn BurnCheckpointConverter,
    ) -> AigcResult<BurnConversionReport> {
        converter
            .convert(source_path, model_id, model_root)
            .map_err(|message| crate::AigcError::BurnCheckpointImport { message })
    }

    pub async fn resolve_readiness(
        &self,
        model_ref: &ModelRef,
    ) -> AigcResult<ModelResolution<ResolvedModelInfo>> {
        let (manifest, _) = self.load_manifest().await?;
        let resolver = ManifestModelResolver::new(&manifest, self.app_paths.models_dir());
        Ok(resolver.resolve_readiness(model_ref).await)
    }

    pub async fn resolve_descriptor(
        &self,
        model_ref: &ModelRef,
    ) -> AigcResult<ModelResolution<ModelDescriptor>> {
        let (manifest, _) = self.load_manifest().await?;
        let resolver = ManifestModelResolver::new(&manifest, self.app_paths.models_dir());
        Ok(resolver.resolve_descriptor(model_ref).await)
    }

    pub async fn resolve_descriptor_with_components(
        &self,
        model_ref: &ModelRef,
    ) -> AigcResult<ModelResolution<ResolvedDescriptorView>> {
        let (manifest, _) = self.load_manifest().await?;
        let resolver = ManifestModelResolver::new(&manifest, self.app_paths.models_dir());
        Ok(resolver.resolve_descriptor_with_components(model_ref).await)
    }

    pub async fn build_readiness_snapshot(
        &self,
        workflow: &reimagine_core::workflow::Workflow,
    ) -> AigcResult<crate::readiness::SnapshotExternalReadinessProvider> {
        use reimagine_core::readiness::ExternalReadinessSubject;
        let (manifest, _) = self.load_manifest().await?;
        let models_dir = self.app_paths.models_dir().to_path_buf();
        let resolver = reimagine_model_manager::ManifestModelResolver::new(&manifest, models_dir);

        let mut provider = crate::readiness::SnapshotExternalReadinessProvider::new();
        for model_ref in collect_model_refs(workflow) {
            let subject = ExternalReadinessSubject::ModelRef(model_ref.clone());
            let resolution = resolver.resolve_readiness(&model_ref).await;
            let diagnostics = resolution.diagnostics().to_vec();
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

    pub fn cached_diagnostics(&self) -> Option<Vec<Diagnostic>> {
        self.diagnostics
            .read()
            .expect("model manifest diagnostics cache poisoned")
            .clone()
    }

    /// Build a `ModelCatalog` from the HuggingFace acquisition config.
    async fn build_catalog(&self) -> AigcResult<ModelCatalog> {
        let cfg = self.acquisition_service.load_config().await?;
        let mut client = hf_hub::HFClientBuilder::new().endpoint(cfg.huggingface.endpoint.clone());
        if let Some(ref token) = cfg.huggingface.token {
            client = client.token(token.clone());
        }
        let hf_client = client.build().map_err(|e| crate::AigcError::Io {
            path: std::path::PathBuf::new(),
            message: format!("failed to build HF client: {e}"),
        })?;
        let mut catalog = ModelCatalog::new(hf_client);
        if let Some(ref token) = cfg.huggingface.token {
            catalog = catalog.with_token(token.clone());
        }
        Ok(catalog)
    }

    /// Search HuggingFace models using the catalog.
    pub async fn search(
        &self,
        query: &ModelSearchQuery,
    ) -> AigcResult<Vec<reimagine_model_acquisition::ModelCatalogEntry>> {
        let catalog = self.build_catalog().await?;
        catalog.search(query).await.map_err(crate::AigcError::from)
    }

    /// Fetch the full model card for a repository.
    pub async fn model_card(&self, repo_id: &str) -> AigcResult<ModelCard> {
        let catalog = self.build_catalog().await?;
        catalog
            .model_card(repo_id)
            .await
            .map_err(crate::AigcError::from)
    }

    fn replace_cached_manifest(&self, manifest: ModelManifest, diagnostics: Vec<Diagnostic>) {
        *self
            .manifest
            .write()
            .expect("model manifest cache poisoned") = Some(manifest);
        *self
            .diagnostics
            .write()
            .expect("model manifest diagnostics cache poisoned") = Some(diagnostics);
    }

    fn normalize_burn_package_report_path(
        &self,
        report_path: &std::path::Path,
    ) -> AigcResult<std::path::PathBuf> {
        let canonical_report =
            report_path
                .canonicalize()
                .map_err(|error| crate::AigcError::Io {
                    path: report_path.to_path_buf(),
                    message: format!("Burn package report path cannot be resolved: {error}"),
                })?;
        let canonical_models_dir = self
            .app_paths
            .models_dir()
            .canonicalize()
            .map_err(|error| crate::AigcError::Io {
                path: self.app_paths.models_dir().to_path_buf(),
                message: format!("models directory cannot be resolved: {error}"),
            })?;

        if !canonical_report.starts_with(&canonical_models_dir) {
            return Err(crate::AigcError::Io {
                path: canonical_report,
                message: "Burn package report path must stay under models directory".to_owned(),
            });
        }

        Ok(report_path.to_path_buf())
    }

    // ------------------------------------------------------------------
    //  Acquire + Convert orchestrator
    // ------------------------------------------------------------------

    /// Download a HuggingFace model, convert it to a backend-native
    /// component layout, and register the result in the manifest.
    pub async fn acquire_and_convert(
        &self,
        request: AcquireAndConvertRequest<'_>,
        acquisition_service: &ModelAcquisitionService,
    ) -> AigcResult<AcquireAndConvertReport> {
        let AcquireAndConvertRequest {
            repo_id,
            model_id,
            revision,
            target_backend,
            overwrite_policy,
            burn_converter,
        } = request;
        if target_backend != "burn" {
            return Err(crate::AigcError::BurnCheckpointImport {
                message: format!("unsupported target_backend `{target_backend}`; expected `burn`"),
            });
        }

        // ---- Step 1: download ----
        let target_relative_dir =
            TargetRelativeDir::new(std::path::PathBuf::from(format!("checkpoints/{model_id}")))
                .map_err(|msg| crate::AigcError::Io {
                    path: self.app_paths.models_dir().to_path_buf(),
                    message: format!("invalid target dir: {msg}"),
                })?;

        let repo = RepoId::new(repo_id).ok_or_else(|| crate::AigcError::Io {
            path: self.app_paths.models_dir().to_path_buf(),
            message: format!("invalid repo_id `{repo_id}`"),
        })?;

        let download_request = ModelAcquisitionRequest {
            provider: AcquireProvider::HuggingFace,
            repo_id: repo.clone(),
            revision: revision.map(Revision::new).unwrap_or_default(),
            allow_patterns: AllowPatterns::new(vec!["*.safetensors".to_string()]),
            target_relative_dir,
            overwrite_policy,
            auto_detect: true,
        };

        // acquire() returns Pin<Box<dyn Future>> when called on Arc<Self>.
        let acq_arc = Arc::new(acquisition_service.clone());
        let acquisition_report = acq_arc.acquire(download_request, None).await?;

        // ---- Step 2: locate the safetensors file ----
        let checkpoint_dir = self
            .app_paths
            .models_dir()
            .join("checkpoints")
            .join(model_id);
        let source_path =
            find_first_safetensors(&checkpoint_dir).ok_or_else(|| crate::AigcError::Io {
                path: checkpoint_dir,
                message: format!("no .safetensors file found in checkpoints/{model_id}"),
            })?;

        // Prepare a cleanup guard that removes the downloaded checkpoint
        // directory if conversion fails before importing.
        let mut cleanup = CleanupGuard::checkpoint(model_id, self.app_paths.models_dir());

        // Burn path: detect diffusers vs single-file format.
        let checkpoint_dir = self
            .app_paths
            .models_dir()
            .join("checkpoints")
            .join(model_id);

        // Check if model_index.json exists — indicates diffusers format
        // where components are already split into subdirectories.
        let diffusers_components = resolve_from_model_index(&checkpoint_dir);

        let (imported_model_id, component_count, source_layout, burn_desc) =
            if let Some(components) = diffusers_components {
                // Diffusers format: components are already split.
                // Build the manifest descriptor directly from the
                // resolved component paths without conversion.
                let imported_model_id = format!("{model_id}-burn");
                let desc = build_diffusers_component_descriptor(
                    &components,
                    &imported_model_id,
                    self.app_paths.models_dir(),
                );
                let component_count = components.len();
                (
                    imported_model_id,
                    component_count,
                    "diffusers_split".to_string(),
                    desc,
                )
            } else {
                // Single-file or CompVis format: convert via Burn port.
                let converter =
                    burn_converter.ok_or_else(|| crate::AigcError::BurnCheckpointImport {
                        message: "no Burn checkpoint converter is configured for this host"
                            .to_owned(),
                    })?;
                let burn_report = self.convert_with_burn_port(
                    &source_path,
                    model_id,
                    self.app_paths.models_dir(),
                    converter,
                )?;

                let imported_model_id = format!("{model_id}-burn");
                let desc = build_burn_component_descriptor(
                    &burn_report,
                    &imported_model_id,
                    self.app_paths.models_dir(),
                );
                let component_count = burn_report.output_components.len();
                (
                    imported_model_id,
                    component_count,
                    burn_report.source_layout,
                    desc,
                )
            };

        let (mut manifest, _) = self.load_manifest().await?;
        manifest.upsert_model(burn_desc);
        self.save_manifest(&manifest).await?;

        cleanup.disarm();

        Ok(AcquireAndConvertReport {
            outcome: "acquired".to_string(),
            model_id: model_id.to_string(),
            imported_model_id: Some(imported_model_id),
            backend: "burn".to_string(),
            mapped_tensor_count: 0,
            component_count,
            source_layout,
            acquisition_report: acquisition_report.repo_id.clone(),
            acquisition_file_count: acquisition_report.files.len(),
            acquisition_total_bytes: acquisition_report.total_bytes,
        })
    }
}

// ======================================================================
//  Free helpers
// ======================================================================

/// A RAII-style guard that removes paths on drop unless disarmed.
struct CleanupGuard {
    paths: Vec<std::path::PathBuf>,
    active: bool,
}

impl CleanupGuard {
    #[allow(dead_code)]
    fn new(paths: Vec<std::path::PathBuf>) -> Self {
        Self {
            paths,
            active: true,
        }
    }

    /// Create a cleanup guard that removes the checkpoint directory.
    fn checkpoint(model_id: &str, models_dir: &std::path::Path) -> Self {
        let path = models_dir.join("checkpoints").join(model_id);
        Self {
            paths: vec![path],
            active: true,
        }
    }

    fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for CleanupGuard {
    fn drop(&mut self) {
        if self.active {
            for path in &self.paths {
                if path.exists() {
                    let _ = std::fs::remove_dir_all(path);
                }
            }
        }
    }
}

/// Walk a directory and return the path to the first `.safetensors` file found.
pub(crate) fn find_first_safetensors(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    if !dir.is_dir() {
        return None;
    }
    for entry in std::fs::read_dir(dir).ok()? {
        let entry = entry.ok()?;
        let path = entry.path();
        if path
            .extension()
            .map(|e| e == "safetensors")
            .unwrap_or(false)
        {
            return Some(path);
        }
    }
    None
}

/// Build a `ModelDescriptor` from a `BurnSdxlConversionReport` using the
/// new flat layout produced by `execute_real_burn_sdxl_checkpoint_import`.
fn build_burn_component_descriptor(
    report: &BurnConversionReport,
    model_id: &str,
    models_dir: &std::path::Path,
) -> ModelDescriptor {
    let components: Vec<ModelComponentSource> = report
        .output_components
        .iter()
        .map(|comp| {
            let role = match comp.role {
                crate::BurnConversionComponentRole::Diffusion => ModelRole::DiffusionModel,
                crate::BurnConversionComponentRole::Vae => ModelRole::Vae,
                crate::BurnConversionComponentRole::TextEncoder
                | crate::BurnConversionComponentRole::TextEncoder2 => ModelRole::TextEncoder,
            };
            // comp.path is relative (e.g. "diffusion_model/<id>.safetensors")
            let path = &comp.path;
            ModelComponentSource::new(
                role,
                ModelSource::relative(
                    ModelRootId::new("base"),
                    path.to_string_lossy().into_owned(),
                ),
                ModelFormat::SafeTensors,
            )
            .with_metadata("component", comp.role.as_str())
            .with_metadata("converted_layout", "burn_native_component")
        })
        .collect();

    ModelDescriptor::new(
        reimagine_core::model::ModelId::new(model_id),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        vec![
            ModelRole::DiffusionModel,
            ModelRole::Vae,
            ModelRole::TextEncoder,
        ],
        ModelSource::relative(
            ModelRootId::new("base"),
            models_dir.to_string_lossy().to_string(),
        ),
        ModelFormat::SafeTensors,
    )
    .with_source_status(ModelSourceStatus::Available)
    .with_components(components)
    .with_metadata("converted_layout", "burn_native_component")
    .with_metadata("source_layout", report.source_layout.clone())
}

/// Build a `ModelDescriptor` from resolved diffusers-format components.
///
/// When a diffusers model is downloaded, the `model_index.json` provides
/// the component mapping. This function converts the resolved component
/// paths into `ModelComponentSource` entries for the manifest, without
/// requiring conversion through the Burn port.
fn build_diffusers_component_descriptor(
    components: &[reimagine_model_acquisition::ResolvedComponent],
    model_id: &str,
    models_dir: &std::path::Path,
) -> ModelDescriptor {
    let model_component_sources: Vec<ModelComponentSource> = components
        .iter()
        .map(|comp| {
            let role = match comp.role {
                ComponentRole::Diffusion => ModelRole::DiffusionModel,
                ComponentRole::Vae => ModelRole::Vae,
                ComponentRole::TextEncoder | ComponentRole::TextEncoder2 => ModelRole::TextEncoder,
            };
            // Compute path relative to models_dir
            let relative_path = comp
                .path
                .strip_prefix(models_dir)
                .unwrap_or(comp.path.as_path())
                .to_string_lossy()
                .replace('\\', "/");

            ModelComponentSource::new(
                role,
                ModelSource::relative(ModelRootId::new("base"), relative_path),
                ModelFormat::SafeTensors,
            )
            .with_metadata("component", comp.role.as_str())
            .with_metadata("source_layout", "diffusers_split")
        })
        .collect();

    // Determine which roles are present
    let mut roles = Vec::new();
    if components
        .iter()
        .any(|c| c.role == ComponentRole::Diffusion)
    {
        roles.push(ModelRole::DiffusionModel);
    }
    if components.iter().any(|c| c.role == ComponentRole::Vae) {
        roles.push(ModelRole::Vae);
    }
    if components
        .iter()
        .any(|c| c.role == ComponentRole::TextEncoder || c.role == ComponentRole::TextEncoder2)
    {
        roles.push(ModelRole::TextEncoder);
    }

    ModelDescriptor::new(
        reimagine_core::model::ModelId::new(model_id),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        roles,
        ModelSource::relative(
            ModelRootId::new("base"),
            models_dir.to_string_lossy().to_string(),
        ),
        ModelFormat::SafeTensors,
    )
    .with_source_status(ModelSourceStatus::Available)
    .with_components(model_component_sources)
    .with_metadata("source_layout", "diffusers_split")
}

// ------------------------------------------------------------------
//  AcquireAndConvertReport — non-serializable internal report struct
// ------------------------------------------------------------------

pub struct AcquireAndConvertRequest<'a> {
    pub repo_id: &'a str,
    pub model_id: &'a str,
    pub revision: Option<&'a str>,
    pub target_backend: &'a str,
    pub overwrite_policy: OverwritePolicy,
    pub burn_converter: Option<&'a dyn BurnCheckpointConverter>,
}

/// Summary returned by [`ModelService::acquire_and_convert`].
pub struct AcquireAndConvertReport {
    pub outcome: String,
    pub model_id: String,
    pub imported_model_id: Option<String>,
    pub backend: String,
    pub mapped_tensor_count: usize,
    pub component_count: usize,
    pub source_layout: String,
    pub acquisition_report: String, // serialized repo_id for the downloaded model
    pub acquisition_file_count: usize,
    pub acquisition_total_bytes: u64,
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
