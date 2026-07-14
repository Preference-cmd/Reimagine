use std::sync::{Arc, RwLock};

use reimagine_config::AppPaths;
use reimagine_core::model::{ModelId, ModelRef, ModelRole, ModelSeries, ModelVariant};
use reimagine_inference_candle::{
    SdxlCheckpointImportRequest, SdxlCheckpointImportResult, SdxlConvertedComponent,
    import_sdxl_checkpoint_to_candle_example_split,
};
use reimagine_model_acquisition::{
    AcquireProvider, AllowPatterns, ModelAcquisitionRequest, OverwritePolicy, RepoId, Revision,
    TargetRelativeDir,
};
use reimagine_model_manager::{
    Fingerprint, ManifestModelResolver, ManifestValidationReport, ModelComponentSource,
    ModelDescriptor, ModelDescriptorResolver, ModelFormat, ModelManifest, ModelManifestStore,
    ModelReadinessResolver, ModelResolution, ModelRootId, ModelSource, ModelSourceStatus,
    ResolvedDescriptorView, ResolvedModelInfo, resolve_source_path, upsert_burn_package_descriptor,
};
use sha2::{Digest, Sha256};

use crate::AppHostResult;
use crate::model_acquisition_service::ModelAcquisitionService;
use crate::{BurnCheckpointConverter, BurnConversionReport};

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

    /// Return the path to the workspace models directory.
    pub fn models_dir(&self) -> &std::path::Path {
        self.app_paths.models_dir()
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

    pub async fn import_sdxl_checkpoint_to_candle_split(
        &self,
        model_id: &ModelId,
    ) -> AppHostResult<(
        ModelManifest,
        ManifestValidationReport,
        SdxlCheckpointImportResult,
    )> {
        let (mut manifest, _) = self.load_manifest().await?;
        let descriptor = manifest
            .models()
            .iter()
            .find(|descriptor| descriptor.id() == model_id)
            .cloned()
            .ok_or_else(|| crate::AppHostError::Io {
                path: self.store.path().to_path_buf(),
                message: format!("model `{model_id}` is not declared in manifest"),
            })?;

        let source_path =
            resolve_source_path(&manifest, descriptor.source(), self.app_paths.models_dir())
                .ok_or_else(|| crate::AppHostError::Io {
                    path: self.store.path().to_path_buf(),
                    message: format!("model `{model_id}` source path cannot be resolved"),
                })?;
        let fingerprint = descriptor
            .fingerprint()
            .ok_or_else(|| crate::AppHostError::Io {
                path: source_path.clone(),
                message: format!(
                    "model `{model_id}` must have a fingerprint before checkpoint import"
                ),
            })?;
        let request = SdxlCheckpointImportRequest::new(
            model_id.as_str(),
            source_path,
            fingerprint_path_segment(fingerprint),
            format!("{:?}", descriptor.format()).to_ascii_lowercase(),
            self.app_paths.models_dir().join("converted"),
        );

        let import_result = import_sdxl_checkpoint_to_candle_example_split(request).await?;
        let converted_descriptor = descriptor_with_converted_components(
            descriptor,
            &import_result,
            self.app_paths.models_dir(),
        );
        manifest.upsert_model(converted_descriptor);
        let report = self.save_manifest(&manifest).await?;
        Ok((manifest, report, import_result))
    }

    pub async fn import_burn_converted_package(
        &self,
        report_path: impl AsRef<std::path::Path>,
    ) -> AppHostResult<(ModelManifest, ManifestValidationReport, ModelDescriptor)> {
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
    ) -> AppHostResult<BurnConversionReport> {
        let (manifest, _) = self.load_manifest().await?;
        let model_id_core = reimagine_core::model::ModelId::new(model_id);
        let descriptor = manifest
            .models()
            .iter()
            .find(|d| d.id() == &model_id_core)
            .cloned()
            .ok_or_else(|| crate::AppHostError::Io {
                path: self.store.path().to_path_buf(),
                message: format!("model `{model_id}` not found in manifest"),
            })?;

        let source_path =
            resolve_source_path(&manifest, descriptor.source(), self.app_paths.models_dir())
                .ok_or_else(|| crate::AppHostError::Io {
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
    ) -> Result<BurnConversionReport, crate::AppHostError> {
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
    ) -> AppHostResult<BurnConversionReport> {
        converter
            .convert(source_path, model_id, model_root)
            .map_err(|message| crate::AppHostError::BurnCheckpointImport { message })
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

    pub async fn resolve_descriptor_with_components(
        &self,
        model_ref: &ModelRef,
    ) -> AppHostResult<ModelResolution<ResolvedDescriptorView>> {
        let (manifest, _) = self.load_manifest().await?;
        let resolver = ManifestModelResolver::new(&manifest, self.app_paths.models_dir());
        Ok(resolver.resolve_descriptor_with_components(model_ref).await)
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

    fn normalize_burn_package_report_path(
        &self,
        report_path: &std::path::Path,
    ) -> AppHostResult<std::path::PathBuf> {
        let canonical_report =
            report_path
                .canonicalize()
                .map_err(|error| crate::AppHostError::Io {
                    path: report_path.to_path_buf(),
                    message: format!("Burn package report path cannot be resolved: {error}"),
                })?;
        let canonical_models_dir = self
            .app_paths
            .models_dir()
            .canonicalize()
            .map_err(|error| crate::AppHostError::Io {
                path: self.app_paths.models_dir().to_path_buf(),
                message: format!("models directory cannot be resolved: {error}"),
            })?;

        if !canonical_report.starts_with(&canonical_models_dir) {
            return Err(crate::AppHostError::Io {
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
    ) -> AppHostResult<AcquireAndConvertReport> {
        let AcquireAndConvertRequest {
            repo_id,
            model_id,
            revision,
            target_backend,
            overwrite_policy,
            burn_converter,
        } = request;
        // ---- Step 1: download ----
        let target_relative_dir =
            TargetRelativeDir::new(std::path::PathBuf::from(format!("checkpoints/{model_id}")))
                .map_err(|msg| crate::AppHostError::Io {
                    path: self.app_paths.models_dir().to_path_buf(),
                    message: format!("invalid target dir: {msg}"),
                })?;

        let repo = RepoId::new(repo_id).ok_or_else(|| crate::AppHostError::Io {
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
            find_first_safetensors(&checkpoint_dir).ok_or_else(|| crate::AppHostError::Io {
                path: checkpoint_dir,
                message: format!("no .safetensors file found in checkpoints/{model_id}"),
            })?;

        // Prepare a cleanup guard that removes the downloaded checkpoint
        // directory if conversion fails before importing.
        let mut cleanup = CleanupGuard::checkpoint(model_id, self.app_paths.models_dir());

        match target_backend {
            "candle" => {
                // Candle path: register checkpoint in manifest, then call
                // import pipeline which does conversion + manifest upsert.
                self.register_checkpoint_descriptor(
                    Some(&acquisition_report),
                    &source_path,
                    model_id,
                )
                .await?;

                let model_id_core = ModelId::new(model_id);
                let (manifest, _report, _import_result) = self
                    .import_sdxl_checkpoint_to_candle_split(&model_id_core)
                    .await?;

                cleanup.disarm();
                let desc = manifest.models().iter().find(|d| d.id() == &model_id_core);

                Ok(AcquireAndConvertReport {
                    outcome: "acquired".to_string(),
                    model_id: model_id.to_string(),
                    imported_model_id: desc.map(|d| d.id().to_string()),
                    backend: "candle".to_string(),
                    mapped_tensor_count: 0,
                    component_count: 0,
                    source_layout: "candle_example_split".to_string(),
                    acquisition_report: acquisition_report.repo_id.clone(),
                    acquisition_file_count: acquisition_report.files.len(),
                    acquisition_total_bytes: acquisition_report.total_bytes,
                })
            }
            _ => {
                // Burn path: convert, build descriptor, upsert manifest.
                let converter =
                    burn_converter.ok_or_else(|| crate::AppHostError::BurnCheckpointImport {
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
                let burn_desc = build_burn_component_descriptor(
                    &burn_report,
                    &imported_model_id,
                    self.app_paths.models_dir(),
                );

                let (mut manifest, _) = self.load_manifest().await?;
                manifest.upsert_model(burn_desc);
                self.save_manifest(&manifest).await?;

                cleanup.disarm();

                let component_count = burn_report.output_components.len();
                Ok(AcquireAndConvertReport {
                    outcome: "acquired".to_string(),
                    model_id: model_id.to_string(),
                    imported_model_id: Some(imported_model_id),
                    backend: "burn".to_string(),
                    mapped_tensor_count: burn_report.mapped_tensor_count,
                    component_count,
                    source_layout: burn_report.source_layout,
                    acquisition_report: acquisition_report.repo_id.clone(),
                    acquisition_file_count: acquisition_report.files.len(),
                    acquisition_total_bytes: acquisition_report.total_bytes,
                })
            }
        }
    }

    /// Register a downloaded checkpoint in the manifest as a
    /// `CheckpointBundle` entry, computing its fingerprint by
    /// hashing the safetensors file.
    async fn register_checkpoint_descriptor(
        &self,
        _acquisition_report: Option<&reimagine_model_acquisition::AcquisitionReport>,
        source_path: &std::path::Path,
        model_id: &str,
    ) -> AppHostResult<ModelDescriptor> {
        let fingerprint = compute_file_sha256(source_path)?;

        let source =
            ModelSource::relative(ModelRootId::new("base"), format!("checkpoints/{model_id}"));

        let descriptor = ModelDescriptor::new(
            ModelId::new(model_id),
            ModelSeries::new("stable_diffusion"),
            ModelVariant::new("sdxl"),
            vec![ModelRole::CheckpointBundle],
            source,
            ModelFormat::Safetensors,
        )
        .with_source_status(ModelSourceStatus::Available)
        .with_fingerprint(Fingerprint::sha256(fingerprint));

        let (mut manifest, _) = self.load_manifest().await?;
        // Check if descriptor already exists — if so skip.
        if manifest.models().iter().any(|d| d.id() == descriptor.id()) {
            let existing = manifest
                .models()
                .iter()
                .find(|d| d.id() == descriptor.id())
                .cloned()
                .unwrap();
            return Ok(existing);
        }
        manifest.upsert_model(descriptor.clone());
        self.save_manifest(&manifest).await?;
        Ok(descriptor)
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

/// Compute the SHA-256 hex digest of a file.
fn compute_file_sha256(path: &std::path::Path) -> AppHostResult<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|e| crate::AppHostError::Io {
        path: path.to_path_buf(),
        message: format!("cannot open for fingerprint: {e}"),
    })?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf).map_err(|e| crate::AppHostError::Io {
            path: path.to_path_buf(),
            message: format!("fingerprint read error: {e}"),
        })?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
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
                ModelFormat::Safetensors,
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
        ModelFormat::Safetensors,
    )
    .with_source_status(ModelSourceStatus::Available)
    .with_components(components)
    .with_metadata("converted_layout", "burn_native_component")
    .with_metadata("source_layout", report.source_layout.clone())
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

fn descriptor_with_converted_components(
    descriptor: ModelDescriptor,
    import_result: &SdxlCheckpointImportResult,
    models_dir: &std::path::Path,
) -> ModelDescriptor {
    let components = SdxlConvertedComponent::all()
        .into_iter()
        .map(|component| component_source(component, import_result, models_dir))
        .collect();

    descriptor
        .with_roles(vec![
            ModelRole::CheckpointBundle,
            ModelRole::DiffusionModel,
            ModelRole::TextEncoder,
            ModelRole::Vae,
        ])
        .with_source_status(ModelSourceStatus::Available)
        .with_metadata("converted_layout", "candle_example_split")
        .with_metadata(
            "converter_version",
            import_result.conversion_manifest().converter_version(),
        )
        .with_components(components)
}

fn component_source(
    component: SdxlConvertedComponent,
    import_result: &SdxlCheckpointImportResult,
    models_dir: &std::path::Path,
) -> ModelComponentSource {
    let role = match component {
        SdxlConvertedComponent::Unet => ModelRole::DiffusionModel,
        SdxlConvertedComponent::Vae => ModelRole::Vae,
        SdxlConvertedComponent::ClipL | SdxlConvertedComponent::ClipG => ModelRole::TextEncoder,
    };
    let path = path_relative_to_models_dir(import_result.component_path(component), models_dir);

    ModelComponentSource::new(
        role,
        ModelSource::relative(ModelRootId::new("base"), path),
        ModelFormat::Safetensors,
    )
    .with_metadata("component", component.metadata_component())
    .with_metadata(
        "converted_layout",
        import_result.conversion_manifest().target_layout(),
    )
}

fn path_relative_to_models_dir(path: std::path::PathBuf, models_dir: &std::path::Path) -> String {
    path.strip_prefix(models_dir)
        .unwrap_or(path.as_path())
        .to_string_lossy()
        .replace('\\', "/")
}

fn fingerprint_path_segment(fingerprint: &Fingerprint) -> String {
    format!(
        "{}-{}",
        sanitize_path_segment(fingerprint.kind()),
        sanitize_path_segment(fingerprint.value())
    )
}

fn sanitize_path_segment(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' => ch,
            _ => '_',
        })
        .collect()
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
