use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use reimagine_core::model::{ModelId, ModelRole};
use reimagine_inference::{
    Backend, BackendInstance, BackendPayloadKey, LoadBundleResponse, ModelFormat, ModelSourceKind,
    ResolvedInferenceModel, ResolvedInferenceModelSource, ResolvedInferenceModelSourceSet,
    RuntimeClipHandle, RuntimeModelHandle, RuntimeVaeHandle,
};
use safetensors::tensor::{Dtype, SafeTensors};

use crate::error::BurnBackendError;

use super::component::{BurnSdxlComponentRole, BurnTensorDType, BurnTensorInventoryEntry};
use super::metadata::BurnComponentMetadata;
use super::validation::{BurnSdxlComponentValidationReport, validate_component_inventory_full};

const PACKAGE_LAYOUT: &str = "burn_native_component_package";
const PACKAGE_CONTRACT: &str = "burn.component";
const PACKAGE_CONTRACT_VERSION: &str = "1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnSdxlSourceSignature {
    components: Vec<BurnSdxlComponentSourceSignature>,
}

impl BurnSdxlSourceSignature {
    fn new(mut components: Vec<BurnSdxlComponentSourceSignature>) -> Self {
        components.sort_by_key(|component| component.role.as_str());
        Self { components }
    }

    /// Empty signature used by test-only metadata builders. Real
    /// bundle signatures are always populated by the bundle loader.
    pub fn empty() -> Self {
        Self {
            components: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BurnSdxlComponentSourceSignature {
    role: BurnSdxlComponentRole,
    path: PathBuf,
    size: u64,
    modified_unix_seconds: Option<u64>,
    contract_version: u32,
}

#[derive(Debug)]
pub enum BurnLoadedModelBundle {
    StableDiffusionSdxl(Arc<BurnLoadedSdxlBundle>),
}

impl BurnLoadedModelBundle {
    pub fn source_signature(&self) -> &BurnSdxlSourceSignature {
        match self {
            Self::StableDiffusionSdxl(bundle) => &bundle.source_signature,
        }
    }

    pub fn load_bundle_response(
        &self,
        backend: Backend,
        backend_instance: BackendInstance,
        device_label: &str,
    ) -> LoadBundleResponse {
        match self {
            Self::StableDiffusionSdxl(bundle) => {
                bundle.load_bundle_response(backend, backend_instance, device_label)
            }
        }
    }
}

#[derive(Debug)]
pub struct BurnLoadedSdxlBundle {
    pub model_id: ModelId,
    pub source_signature: BurnSdxlSourceSignature,
    pub diffusion_payload_key: BackendPayloadKey,
    pub clip_payload_key: BackendPayloadKey,
    pub vae_payload_key: BackendPayloadKey,
    #[allow(dead_code)]
    pub components: Vec<BurnLoadedSdxlComponent>,
}

impl BurnLoadedSdxlBundle {
    pub fn from_resolved(
        resolved: &ResolvedInferenceModel,
        source_set: &ResolvedInferenceModelSourceSet,
    ) -> Result<Self, BurnBackendError> {
        if resolved.series().as_str() != "stable_diffusion" || resolved.variant().as_str() != "sdxl"
        {
            return Err(BurnBackendError::UnsupportedSourceLayout(format!(
                "Burn load_bundle only supports stable_diffusion/sdxl, got {}/{} for model `{}`",
                resolved.series(),
                resolved.variant(),
                resolved.model_id()
            )));
        }

        if resolved.role() != ModelRole::CheckpointBundle {
            return Err(BurnBackendError::UnsupportedSourceLayout(format!(
                "Burn load_bundle expects resolved model role CheckpointBundle for model `{}`",
                resolved.model_id()
            )));
        }

        let components = resolve_components(resolved.model_id(), source_set)?;
        let signature = BurnSdxlSourceSignature::new(
            components
                .iter()
                .map(|component| component.source_signature.clone())
                .collect(),
        );

        let model_id = resolved.model_id().clone();
        let key_prefix = format!("burn:model:{model_id}");

        Ok(Self {
            model_id,
            source_signature: signature,
            diffusion_payload_key: BackendPayloadKey::new(format!("{key_prefix}:diffusion")),
            clip_payload_key: BackendPayloadKey::new(format!("{key_prefix}:clip")),
            vae_payload_key: BackendPayloadKey::new(format!("{key_prefix}:vae")),
            components,
        })
    }

    pub fn source_signature(&self) -> &BurnSdxlSourceSignature {
        &self.source_signature
    }

    /// Return the loaded SDXL components (weight file sources and
    /// metadata) in this bundle.
    pub fn components(&self) -> &[BurnLoadedSdxlComponent] {
        &self.components
    }

    /// Borrow the model id this bundle was loaded for. Used by
    /// the cross-run cache and by the text-encode preflight to
    /// record the conditioning payload's provenance.
    pub fn model_id(&self) -> &ModelId {
        &self.model_id
    }

    /// Return the paths of the primary (text_encoder) and secondary
    /// (text_encoder_2) component files in this bundle.
    pub fn text_encoder_component_paths(&self) -> Result<(PathBuf, PathBuf), BurnBackendError> {
        let primary = self
            .components
            .iter()
            .find(|c| c.component_role == BurnSdxlComponentRole::TextEncoder)
            .ok_or_else(|| BurnBackendError::MissingComponent("text_encoder".to_owned()))?;
        let secondary = self
            .components
            .iter()
            .find(|c| c.component_role == BurnSdxlComponentRole::TextEncoder2)
            .ok_or_else(|| BurnBackendError::MissingComponent("text_encoder_2".to_owned()))?;
        Ok((primary.source_path.clone(), secondary.source_path.clone()))
    }

    pub(crate) fn uses_tiny_sdxl_e2e_text_profiles(&self) -> bool {
        let primary = self
            .components
            .iter()
            .find(|c| c.component_role == BurnSdxlComponentRole::TextEncoder);
        let secondary = self
            .components
            .iter()
            .find(|c| c.component_role == BurnSdxlComponentRole::TextEncoder2);

        matches!(
            (primary, secondary),
            (Some(primary), Some(secondary))
                if primary.is_tiny_sdxl_e2e_fixture()
                    && secondary.is_tiny_sdxl_e2e_fixture()
        )
    }

    pub(crate) fn uses_tiny_sdxl_e2e_diffusion_profile(&self) -> bool {
        self.components
            .iter()
            .find(|c| c.component_role == BurnSdxlComponentRole::Diffusion)
            .is_some_and(BurnLoadedSdxlComponent::is_tiny_sdxl_e2e_fixture)
    }

    pub(crate) fn uses_tiny_sdxl_e2e_vae_profile(&self) -> bool {
        self.components
            .iter()
            .find(|c| c.component_role == BurnSdxlComponentRole::Vae)
            .is_some_and(BurnLoadedSdxlComponent::is_tiny_sdxl_e2e_fixture)
    }

    /// Test-only constructor that builds a minimal bundle for
    /// the cross-run cache without going through the file-system
    /// resolver. Real production code must use
    /// [`BurnLoadedSdxlBundle::from_resolved`].
    #[cfg(test)]
    pub(crate) fn for_test_only(model_id: ModelId, clip_payload_key: BackendPayloadKey) -> Self {
        let key_prefix = format!("burn:model:{model_id}");
        Self {
            model_id,
            source_signature: BurnSdxlSourceSignature::empty(),
            diffusion_payload_key: BackendPayloadKey::new(format!("{key_prefix}:diffusion")),
            clip_payload_key,
            vae_payload_key: BackendPayloadKey::new(format!("{key_prefix}:vae")),
            components: Vec::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn with_test_components(
        mut self,
        components: Vec<(BurnSdxlComponentRole, PathBuf)>,
    ) -> Self {
        self.components = components
            .into_iter()
            .map(|(component_role, source_path)| {
                BurnLoadedSdxlComponent::for_test_only(component_role, source_path)
            })
            .collect();
        self.source_signature = BurnSdxlSourceSignature::new(
            self.components
                .iter()
                .map(|component| component.source_signature.clone())
                .collect(),
        );
        self
    }

    #[cfg(test)]
    pub(crate) fn with_test_tiny_fixture_components(
        mut self,
        components: Vec<(BurnSdxlComponentRole, PathBuf)>,
    ) -> Self {
        self.components = components
            .into_iter()
            .map(|(component_role, source_path)| {
                BurnLoadedSdxlComponent::for_test_only_tiny_fixture(component_role, source_path)
            })
            .collect();
        self.source_signature = BurnSdxlSourceSignature::new(
            self.components
                .iter()
                .map(|component| component.source_signature.clone())
                .collect(),
        );
        self
    }

    fn load_bundle_response(
        &self,
        backend: Backend,
        backend_instance: BackendInstance,
        device_label: &str,
    ) -> LoadBundleResponse {
        let model = RuntimeModelHandle::with_instance(
            self.model_id.clone(),
            ModelRole::CheckpointBundle,
            backend.clone(),
            backend_instance.clone(),
            self.diffusion_payload_key.clone(),
        )
        .with_device(device_label);
        let clip = RuntimeClipHandle::with_instance(
            self.model_id.clone(),
            backend.clone(),
            backend_instance.clone(),
            self.clip_payload_key.clone(),
        )
        .with_device(device_label);
        let vae = RuntimeVaeHandle::with_instance(
            self.model_id.clone(),
            backend,
            backend_instance,
            self.vae_payload_key.clone(),
        )
        .with_device(device_label);

        LoadBundleResponse::new(model, clip, vae)
    }
}

#[derive(Debug)]
pub struct BurnLoadedSdxlComponent {
    pub component_role: BurnSdxlComponentRole,
    pub source_path: PathBuf,
    #[allow(dead_code)]
    metadata: BurnComponentMetadata,
    #[allow(dead_code)]
    validation_report: BurnSdxlComponentValidationReport,
    #[allow(dead_code)]
    inventory: Vec<BurnTensorInventoryEntry>,
    source_signature: BurnSdxlComponentSourceSignature,
}

impl BurnLoadedSdxlComponent {
    #[cfg(test)]
    fn for_test_only(component_role: BurnSdxlComponentRole, source_path: PathBuf) -> Self {
        let metadata = BurnComponentMetadata {
            contract: super::contract::CONTRACT_NAME.to_owned(),
            component_role,
            contract_version: 1,
            backend: super::contract::BACKEND_NAME.to_owned(),
            model_series: super::contract::MODEL_SERIES.to_owned(),
            variant: super::contract::VARIANT.to_owned(),
            tensor_layout: super::contract::TENSOR_LAYOUT.to_owned(),
            dtype_policy: super::contract::BurnDTypePolicy::Fp32,
            fixture_profile: None,
        };
        Self {
            component_role,
            source_path: source_path.clone(),
            metadata,
            validation_report: BurnSdxlComponentValidationReport {
                component_role,
                matched_required_tensors: Vec::new(),
                missing_required_tensors: Vec::new(),
                unused_tensors: Vec::new(),
                warnings: Vec::new(),
            },
            inventory: Vec::new(),
            source_signature: BurnSdxlComponentSourceSignature {
                role: component_role,
                path: source_path,
                size: 0,
                modified_unix_seconds: None,
                contract_version: 1,
            },
        }
    }

    #[cfg(test)]
    fn for_test_only_tiny_fixture(
        component_role: BurnSdxlComponentRole,
        source_path: PathBuf,
    ) -> Self {
        let mut component = Self::for_test_only(component_role, source_path);
        component.metadata.fixture_profile =
            Some(super::metadata::metadata_keys::TINY_SDXL_E2E_PROFILE.to_owned());
        component
    }

    pub(crate) fn is_tiny_sdxl_e2e_fixture(&self) -> bool {
        self.metadata.is_tiny_sdxl_e2e_fixture()
    }
}

fn resolve_components(
    model_id: &ModelId,
    source_set: &ResolvedInferenceModelSourceSet,
) -> Result<Vec<BurnLoadedSdxlComponent>, BurnBackendError> {
    if source_set.is_checkpoint_bundle()
        || source_set
            .sources()
            .iter()
            .any(|source| source.kind() == ModelSourceKind::CheckpointBundle)
    {
        return Err(BurnBackendError::UnsupportedSourceLayout(format!(
            "Burn model `{model_id}` requires burn/04 converted SplitComponent sources; checkpoint bundle sources must be converted before runtime load"
        )));
    }

    if source_set.sources().len() != BurnSdxlComponentRole::all().len() {
        return Err(BurnBackendError::UnsupportedSourceLayout(format!(
            "Burn model `{model_id}` requires exactly 4 converted SplitComponent sources, found {}",
            source_set.sources().len()
        )));
    }

    let mut components = Vec::new();
    let mut seen = Vec::new();

    for source in source_set.sources() {
        let component = inspect_source(model_id, source)?;
        let role = component.component_role;
        if seen.contains(&role) {
            return Err(BurnBackendError::DuplicateComponent(
                role.as_str().to_owned(),
            ));
        }
        seen.push(role);
        components.push(component);
    }

    let missing = BurnSdxlComponentRole::all()
        .into_iter()
        .filter(|role| !seen.contains(role))
        .map(|role| role.as_str())
        .collect::<Vec<_>>();

    if !missing.is_empty() {
        return Err(BurnBackendError::MissingComponent(missing.join(", ")));
    }

    Ok(components)
}

fn inspect_source(
    model_id: &ModelId,
    source: &ResolvedInferenceModelSource,
) -> Result<BurnLoadedSdxlComponent, BurnBackendError> {
    if source.kind() != ModelSourceKind::SplitComponent {
        return Err(BurnBackendError::UnsupportedSourceLayout(format!(
            "Burn model `{model_id}` requires SplitComponent sources, found {:?} for `{}`",
            source.kind(),
            source.path().display()
        )));
    }
    if source.format() != ModelFormat::SafeTensors {
        return Err(BurnBackendError::UnsupportedSourceLayout(format!(
            "Burn model `{model_id}` requires safetensors components, found {:?} for `{}`",
            source.format(),
            source.path().display()
        )));
    }

    let projection = source
        .metadata()
        .map(parse_projection_metadata)
        .transpose()?;
    validate_projection_metadata(model_id, source, projection.as_ref())?;

    let inspected = inspect_component_safetensors(source.path())?;
    let metadata = BurnComponentMetadata::parse(&inspected.metadata).map_err(|source_error| {
        BurnBackendError::ComponentValidation {
            path: source.path().clone(),
            source: source_error,
        }
    })?;
    let validation_report =
        validate_component_inventory_full(&inspected.metadata, &inspected.inventory).map_err(
            |source_error| BurnBackendError::ComponentValidation {
                path: source.path().clone(),
                source: source_error,
            },
        )?;

    if let Some(projection) = projection.as_ref()
        && let Some(component) = projection.get("component")
        && component != metadata.component_role.as_str()
    {
        return Err(BurnBackendError::ComponentMetadataMismatch {
            path: source.path().clone(),
            expected: format!("component={component}"),
            found: format!("component={}", metadata.component_role.as_str()),
        });
    }

    validate_role_pair(model_id, source, metadata.component_role)?;

    let file_signature = component_signature(source.path(), &metadata)?;

    Ok(BurnLoadedSdxlComponent {
        component_role: metadata.component_role,
        source_path: source.path().clone(),
        metadata,
        validation_report,
        inventory: inspected.inventory,
        source_signature: file_signature,
    })
}

fn validate_projection_metadata(
    model_id: &ModelId,
    source: &ResolvedInferenceModelSource,
    projection: Option<&BTreeMap<String, String>>,
) -> Result<(), BurnBackendError> {
    let Some(projection) = projection else {
        return Ok(());
    };

    let expected = [
        ("backend", "burn"),
        ("converted_layout", PACKAGE_LAYOUT),
        ("contract", PACKAGE_CONTRACT),
        ("contract_version", PACKAGE_CONTRACT_VERSION),
    ];

    for (key, expected_value) in expected {
        if let Some(found) = projection.get(key)
            && found != expected_value
        {
            return Err(BurnBackendError::ComponentMetadataMismatch {
                path: source.path().clone(),
                expected: format!("{key}={expected_value}"),
                found: format!("{key}={found}"),
            });
        }
    }

    if let Some(component) = projection.get("component") {
        BurnSdxlComponentRole::try_from(component.as_str()).map_err(|_| {
            BurnBackendError::UnsupportedSourceLayout(format!(
                "Burn model `{model_id}` has unsupported component projection `{component}` for `{}`",
                source.path().display()
            ))
        })?;
    }

    Ok(())
}

fn validate_role_pair(
    model_id: &ModelId,
    source: &ResolvedInferenceModelSource,
    component_role: BurnSdxlComponentRole,
) -> Result<(), BurnBackendError> {
    let expected = match component_role {
        BurnSdxlComponentRole::Diffusion => ModelRole::DiffusionModel,
        BurnSdxlComponentRole::Vae => ModelRole::Vae,
        BurnSdxlComponentRole::TextEncoder | BurnSdxlComponentRole::TextEncoder2 => {
            ModelRole::TextEncoder
        }
    };

    if source.role() != expected {
        return Err(BurnBackendError::UnsupportedSourceLayout(format!(
            "Burn model `{model_id}` component `{}` expects source role {:?}, found {:?} for `{}`",
            component_role.as_str(),
            expected,
            source.role(),
            source.path().display()
        )));
    }

    Ok(())
}

fn parse_projection_metadata(raw: &str) -> Result<BTreeMap<String, String>, BurnBackendError> {
    let mut parsed = BTreeMap::new();
    for entry in raw
        .split(';')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
    {
        let Some((key, value)) = entry.split_once('=') else {
            return Err(BurnBackendError::InvalidRequest(format!(
                "invalid Burn resolver metadata entry `{entry}`"
            )));
        };
        parsed.insert(key.trim().to_owned(), value.trim().to_owned());
    }
    Ok(parsed)
}

#[derive(Debug)]
struct InspectedComponent {
    metadata: BTreeMap<String, String>,
    inventory: Vec<BurnTensorInventoryEntry>,
}

fn inspect_component_safetensors(path: &Path) -> Result<InspectedComponent, BurnBackendError> {
    let file_type = fs::metadata(path).map_err(|source| BurnBackendError::ComponentRead {
        path: path.to_path_buf(),
        message: source.to_string(),
    })?;

    if !file_type.is_file() {
        return Err(BurnBackendError::ComponentRead {
            path: path.to_path_buf(),
            message: "component path is not a file".to_owned(),
        });
    }

    let bytes = fs::read(path).map_err(|source| BurnBackendError::ComponentRead {
        path: path.to_path_buf(),
        message: source.to_string(),
    })?;
    let (_, file_metadata) =
        SafeTensors::read_metadata(&bytes).map_err(|source| BurnBackendError::ComponentRead {
            path: path.to_path_buf(),
            message: source.to_string(),
        })?;
    let metadata = file_metadata
        .metadata()
        .clone()
        .unwrap_or_default()
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let safetensors =
        SafeTensors::deserialize(&bytes).map_err(|source| BurnBackendError::ComponentRead {
            path: path.to_path_buf(),
            message: source.to_string(),
        })?;
    let inventory = safetensors
        .names()
        .into_iter()
        .map(|name| {
            let tensor =
                safetensors
                    .tensor(name)
                    .map_err(|source| BurnBackendError::ComponentRead {
                        path: path.to_path_buf(),
                        message: source.to_string(),
                    })?;
            Ok(BurnTensorInventoryEntry::new(
                name.to_owned(),
                tensor.shape().to_vec(),
                burn_dtype(tensor.dtype()),
            ))
        })
        .collect::<Result<Vec<_>, BurnBackendError>>()?;

    Ok(InspectedComponent {
        metadata,
        inventory,
    })
}

fn component_signature(
    path: &Path,
    metadata: &BurnComponentMetadata,
) -> Result<BurnSdxlComponentSourceSignature, BurnBackendError> {
    let file_metadata = fs::metadata(path).map_err(|source| BurnBackendError::ComponentRead {
        path: path.to_path_buf(),
        message: source.to_string(),
    })?;
    let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let modified_unix_seconds = file_metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs());

    Ok(BurnSdxlComponentSourceSignature {
        role: metadata.component_role,
        path,
        size: file_metadata.len(),
        modified_unix_seconds,
        contract_version: metadata.contract_version,
    })
}

fn burn_dtype(dtype: Dtype) -> BurnTensorDType {
    match dtype {
        Dtype::F32 => BurnTensorDType::F32,
        Dtype::F16 => BurnTensorDType::F16,
        Dtype::BF16 => BurnTensorDType::Bf16,
        other => BurnTensorDType::Unsupported(format!("{other:?}")),
    }
}
