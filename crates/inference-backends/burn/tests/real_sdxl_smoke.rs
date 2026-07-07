use std::path::PathBuf;
use std::sync::Arc;

use reimagine_core::model::{
    ModelId, ModelRole, ModelSeries, ModelVariant, NodeId, ParamValue, RunId, WorkflowId,
    WorkflowVersion,
};
use reimagine_inference::{
    CreateEmptyLatentRequest, DiffusionSampleRequest, ExecutionValue, ImagePreviewRequest,
    ImageSaveRequest, InferenceBackend, LatentDecodeRequest, LoadBundleRequest, ModelFormat,
    ModelSourceKind, ResolvedInferenceModel, ResolvedInferenceModelSource,
    ResolvedInferenceModelSourceSet, SamplerName, SchedulerName, TextEncodeRequest,
};
use reimagine_inference_burn::models::stable_diffusion::sdxl::BurnSdxlComponentRole;
use reimagine_inference_burn::{BurnBackend, BurnBackendConfig, BurnDevice};

const PACKAGE_ROOT_ENV: &str = "REIMAGINE_BURN_REAL_SDXL_PACKAGE";
const MODEL_ID_ENV: &str = "REIMAGINE_BURN_REAL_SDXL_MODEL_ID";
const STEPS_ENV: &str = "REIMAGINE_BURN_REAL_SDXL_STEPS";
const SEED_ENV: &str = "REIMAGINE_BURN_REAL_SDXL_SEED";
const DEVICE_ENV: &str = "REIMAGINE_BURN_REAL_SDXL_DEVICE";
const PROMPT_ENV: &str = "REIMAGINE_BURN_REAL_SDXL_PROMPT";
const NEGATIVE_PROMPT_ENV: &str = "REIMAGINE_BURN_REAL_SDXL_NEGATIVE_PROMPT";
const CONVERTER_VERSION_MARKER: &str = "burn-sdxl-package-15f-v1";
const RUN_ID: &str = "run-burn-real-sdxl-smoke";
const WORKFLOW_ID: &str = "wf-burn-real-sdxl-smoke";

#[derive(Debug, Clone, PartialEq, Eq)]
enum RealSdxlSmokeConfig {
    Skipped {
        reason: String,
    },
    Enabled {
        package_root: PathBuf,
        model_id: String,
        steps: u32,
        seed: u64,
        device_label: Option<String>,
        prompt: String,
        negative_prompt: String,
    },
}

#[derive(serde::Deserialize)]
struct ConversionReportView {
    package: Option<PackageReportView>,
}

#[derive(serde::Deserialize)]
struct PackageReportView {
    converter_version: String,
}

impl RealSdxlSmokeConfig {
    fn from_env_getter(get: impl Fn(&str) -> Option<String>) -> Self {
        match get(PACKAGE_ROOT_ENV).filter(|value| !value.trim().is_empty()) {
            Some(package_root) => Self::Enabled {
                package_root: PathBuf::from(package_root),
                model_id: get(MODEL_ID_ENV)
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or_else(|| "burn-real-sdxl-smoke".to_owned()),
                steps: get(STEPS_ENV)
                    .and_then(|value| value.parse::<u32>().ok())
                    .filter(|steps| *steps > 0)
                    .unwrap_or(1),
                seed: get(SEED_ENV)
                    .and_then(|value| value.parse::<u64>().ok())
                    .unwrap_or(1234),
                device_label: get(DEVICE_ENV).filter(|value| !value.trim().is_empty()),
                prompt: get(PROMPT_ENV)
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or_else(|| "small bright city at sunrise".to_owned()),
                negative_prompt: get(NEGATIVE_PROMPT_ENV)
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or_else(|| "low quality blur".to_owned()),
            },
            None => Self::Skipped {
                reason: format!("set {PACKAGE_ROOT_ENV} to a converted Burn SDXL package root"),
            },
        }
    }
}

#[test]
fn real_sdxl_smoke_is_explicitly_gated_by_package_env() {
    let config = RealSdxlSmokeConfig::from_env_getter(|_| None);

    assert_eq!(
        config,
        RealSdxlSmokeConfig::Skipped {
            reason: format!("set {PACKAGE_ROOT_ENV} to a converted Burn SDXL package root")
        }
    );
}

#[test]
fn real_sdxl_smoke_builds_split_component_model_from_package_root() {
    let config = RealSdxlSmokeConfig::from_env_getter(|key| match key {
        PACKAGE_ROOT_ENV => Some("/models/converted/burn/sdxl-base/fingerprint".to_owned()),
        MODEL_ID_ENV => Some("local-sdxl-base".to_owned()),
        _ => None,
    });
    let RealSdxlSmokeConfig::Enabled {
        package_root,
        model_id,
        ..
    } = config
    else {
        panic!("expected enabled smoke config");
    };

    let model = resolved_model_from_package(&package_root, &model_id);

    assert_eq!(model.model_id().as_str(), "local-sdxl-base");
    assert_eq!(model.series(), &ModelSeries::new("stable_diffusion"));
    assert_eq!(model.variant(), &ModelVariant::new("sdxl"));
    let sources = model
        .source_set()
        .expect("smoke model has source set")
        .sources();
    assert_eq!(sources.len(), 4);
    for role in BurnSdxlComponentRole::all() {
        let source = sources
            .iter()
            .find(|source| {
                source
                    .path()
                    .ends_with(format!("{}/model.safetensors", role.as_str()))
            })
            .unwrap_or_else(|| panic!("missing source for role {role}"));
        assert_eq!(source.kind(), ModelSourceKind::SplitComponent);
        assert_eq!(source.format(), ModelFormat::SafeTensors);
        assert!(
            source
                .metadata()
                .expect("source metadata")
                .contains(&format!("component={}", role.as_str()))
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RealSdxlSmokeStage {
    PackageValidation,
    LoadBundle,
    TextEncode,
    LatentCreate,
    Sampler,
    VaeDecode,
    Preview,
    Save,
}

impl RealSdxlSmokeStage {
    const fn as_str(self) -> &'static str {
        match self {
            Self::PackageValidation => "package_validation",
            Self::LoadBundle => "load_bundle",
            Self::TextEncode => "text_encode",
            Self::LatentCreate => "latent_create",
            Self::Sampler => "sampler",
            Self::VaeDecode => "vae_decode",
            Self::Preview => "preview",
            Self::Save => "save",
        }
    }
}

fn stage_context<T, E: std::fmt::Display>(
    stage: RealSdxlSmokeStage,
    result: Result<T, E>,
) -> Result<T, String> {
    result.map_err(|err| format!("real SDXL smoke failed at {}: {err}", stage.as_str()))
}

#[test]
fn real_sdxl_smoke_stage_errors_name_failed_stage() {
    let err = stage_context(
        RealSdxlSmokeStage::LoadBundle,
        Err::<(), _>("component contract mismatch".to_owned()),
    )
    .expect_err("stage should fail");

    assert_eq!(
        err,
        "real SDXL smoke failed at load_bundle: component contract mismatch"
    );
}

#[test]
fn real_sdxl_smoke_rejects_stale_package_converter_version() {
    let root = tempfile::tempdir().expect("package root");
    std::fs::write(
        root.path().join("conversion-report.json"),
        r#"{"package":{"converter_version":"old-version"}}"#,
    )
    .expect("write report");

    let err = validate_package_root(root.path()).expect_err("stale package should fail");

    assert!(
        err.contains("expected converter version `burn-sdxl-package-15f-v1`"),
        "{err}"
    );
}

#[tokio::test]
async fn real_sdxl_component_package_runs_public_burn_capability_chain_when_enabled() {
    let config = RealSdxlSmokeConfig::from_env_getter(|key| std::env::var(key).ok());
    let RealSdxlSmokeConfig::Enabled {
        package_root,
        model_id,
        steps,
        seed,
        device_label,
        prompt,
        negative_prompt,
    } = config
    else {
        eprintln!(
            "skipping real SDXL Burn smoke: set {PACKAGE_ROOT_ENV} to a converted Burn SDXL package root"
        );
        return;
    };

    let output = tempfile::tempdir().expect("smoke output dir");
    let mut backend_config = BurnBackendConfig::new(&package_root, output.path());
    if let Some(device_label) = device_label.as_deref() {
        backend_config = backend_config.with_device(
            BurnDevice::try_build_device(device_label).expect("valid real smoke burn device label"),
        );
    }
    let backend = BurnBackend::new(backend_config).expect("burn backend");

    stage_context(
        RealSdxlSmokeStage::PackageValidation,
        validate_package_root(&package_root),
    )
    .expect("real package validation");

    let model = resolved_model_from_package(&package_root, &model_id);
    eprintln!(
        "real SDXL Burn smoke: package_root={}, backend_device={}, converter_marker={}, steps={}, seed={}, sampler=euler, scheduler=normal",
        package_root.display(),
        backend.config().device_label(),
        CONVERTER_VERSION_MARKER,
        steps,
        seed
    );

    let loaded = stage_context(
        RealSdxlSmokeStage::LoadBundle,
        backend.load_bundle(load_request(model)).await,
    )
    .expect("load_bundle");
    let positive = stage_context(
        RealSdxlSmokeStage::TextEncode,
        backend
            .text_encode(text_request(
                loaded.clip().clone(),
                &prompt,
                "text-positive",
            ))
            .await,
    )
    .expect("positive text.encode")
    .into_conditioning();
    let negative = stage_context(
        RealSdxlSmokeStage::TextEncode,
        backend
            .text_encode(text_request(
                loaded.clip().clone(),
                &negative_prompt,
                "text-negative",
            ))
            .await,
    )
    .expect("negative text.encode")
    .into_conditioning();

    let latent = stage_context(
        RealSdxlSmokeStage::LatentCreate,
        backend
            .create_empty_latent(CreateEmptyLatentRequest::new(
                1024,
                1024,
                1,
                run_id(),
                workflow_id(),
                workflow_version(),
                NodeId::new("latent-empty"),
            ))
            .await,
    )
    .expect("latent.create_empty")
    .into_latent();
    let sampled = stage_context(
        RealSdxlSmokeStage::Sampler,
        backend
            .diffusion_sample(DiffusionSampleRequest::new(
                loaded.model().clone(),
                positive,
                negative,
                latent,
                seed,
                steps,
                1.0,
                SamplerName::Euler,
                SchedulerName::Normal,
                1.0,
                run_id(),
                workflow_id(),
                workflow_version(),
                NodeId::new("diffusion"),
            ))
            .await,
    )
    .expect("diffusion.sample")
    .into_latent();
    let image = stage_context(
        RealSdxlSmokeStage::VaeDecode,
        backend
            .latent_decode(LatentDecodeRequest::new(
                loaded.vae().clone(),
                sampled,
                run_id(),
                workflow_id(),
                workflow_version(),
                NodeId::new("vae-decode"),
            ))
            .await,
    )
    .expect("latent.decode")
    .into_image();

    let preview = stage_context(
        RealSdxlSmokeStage::Preview,
        backend
            .image_preview(ImagePreviewRequest::new(
                image.clone(),
                run_id(),
                workflow_id(),
                workflow_version(),
                NodeId::new("preview"),
            ))
            .await,
    )
    .expect("image.preview")
    .into_artifact();
    let saved = stage_context(
        RealSdxlSmokeStage::Save,
        backend
            .image_save(
                ImageSaveRequest::new(
                    image,
                    run_id(),
                    workflow_id(),
                    workflow_version(),
                    NodeId::new("save"),
                )
                .with_filename_prefix("burn-real-sdxl"),
            )
            .await,
    )
    .expect("image.save")
    .into_artifact();

    assert_png_artifact(output.path(), preview.as_str());
    assert_png_artifact(output.path(), saved.as_str());
    eprintln!(
        "real SDXL Burn smoke produced preview={} saved={}",
        preview.as_str(),
        saved.as_str()
    );
}

fn resolved_model_from_package(
    package_root: &std::path::Path,
    model_id: &str,
) -> ResolvedInferenceModel {
    let sources = BurnSdxlComponentRole::all()
        .into_iter()
        .map(|role| split_component_source(package_root, role))
        .collect();

    ResolvedInferenceModel::new(
        ModelId::new(model_id),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        package_root.to_path_buf(),
        ModelFormat::SafeTensors,
    )
    .with_source_set(ResolvedInferenceModelSourceSet::from_sources(sources))
}

fn load_request(model: ResolvedInferenceModel) -> LoadBundleRequest {
    LoadBundleRequest::new(
        model,
        run_id(),
        workflow_id(),
        workflow_version(),
        NodeId::new("load-bundle"),
    )
}

fn text_request(
    clip: reimagine_inference::RuntimeClipHandle,
    prompt: &str,
    node: &str,
) -> TextEncodeRequest {
    TextEncodeRequest::new(
        clip,
        Arc::new(ExecutionValue::Param(ParamValue::String(prompt.to_owned()))),
        run_id(),
        workflow_id(),
        workflow_version(),
        NodeId::new(node),
    )
}

fn validate_package_root(package_root: &std::path::Path) -> Result<(), String> {
    if !package_root.is_dir() {
        return Err(format!(
            "package root `{}` is not a directory",
            package_root.display()
        ));
    }
    let report_path = package_root.join("conversion-report.json");
    let report_json = std::fs::read_to_string(&report_path)
        .map_err(|err| format!("read `{}`: {err}", report_path.display()))?;
    let report: ConversionReportView = serde_json::from_str(&report_json)
        .map_err(|err| format!("parse `{}`: {err}", report_path.display()))?;
    let converter_version = report
        .package
        .as_ref()
        .map(|package| package.converter_version.as_str())
        .ok_or_else(|| format!("`{}` is missing package metadata", report_path.display()))?;
    if converter_version != CONVERTER_VERSION_MARKER {
        return Err(format!(
            "`{}` has converter version `{}`, expected converter version `{}`",
            report_path.display(),
            converter_version,
            CONVERTER_VERSION_MARKER
        ));
    }
    for role in BurnSdxlComponentRole::all() {
        let component = package_root.join(role.as_str()).join("model.safetensors");
        if !component.is_file() {
            return Err(format!(
                "missing component `{}` at `{}`",
                role,
                component.display()
            ));
        }
    }
    Ok(())
}

fn assert_png_artifact(output_root: &std::path::Path, artifact: &str) {
    assert!(
        artifact.ends_with(".png"),
        "artifact should be png: {artifact}"
    );
    let path = output_root.join(artifact);
    assert!(path.is_file(), "artifact file exists: {}", path.display());
    let bytes = std::fs::read(path).expect("read artifact");
    assert!(
        bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
        "artifact has PNG signature"
    );
}

fn run_id() -> RunId {
    RunId::new(RUN_ID)
}

fn workflow_id() -> WorkflowId {
    WorkflowId::new(WORKFLOW_ID)
}

fn workflow_version() -> WorkflowVersion {
    WorkflowVersion::new(1)
}

fn split_component_source(
    package_root: &std::path::Path,
    role: BurnSdxlComponentRole,
) -> ResolvedInferenceModelSource {
    ResolvedInferenceModelSource::new(
        ModelSourceKind::SplitComponent,
        role_model_role(role),
        package_root.join(role.as_str()).join("model.safetensors"),
        ModelFormat::SafeTensors,
    )
    .with_metadata(format!(
        "component={};backend=burn;converted_layout=burn_native_component_package;contract=burn.component;contract_version=1",
        role.as_str()
    ))
}

fn role_model_role(role: BurnSdxlComponentRole) -> ModelRole {
    match role {
        BurnSdxlComponentRole::Diffusion => ModelRole::DiffusionModel,
        BurnSdxlComponentRole::Vae => ModelRole::Vae,
        BurnSdxlComponentRole::TextEncoder | BurnSdxlComponentRole::TextEncoder2 => {
            ModelRole::TextEncoder
        }
    }
}
