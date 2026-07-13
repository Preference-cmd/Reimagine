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
use reimagine_inference_burn::models::stable_diffusion::sdxl::{
    BurnSdxlComponentRole, BurnSdxlDiffusersSplitPackageRequest,
    package_diffusers_style_split_sdxl_source,
};
use reimagine_inference_burn::{BurnBackend, BurnBackendConfig, BurnDevice};

const PACKAGE_ROOT_ENV: &str = "REIMAGINE_BURN_REAL_SDXL_PACKAGE";
const SPLIT_SOURCE_ROOT_ENV: &str = "REIMAGINE_BURN_REAL_SDXL_SPLIT_SOURCE";
const MODEL_ID_ENV: &str = "REIMAGINE_BURN_REAL_SDXL_MODEL_ID";
const STEPS_ENV: &str = "REIMAGINE_BURN_REAL_SDXL_STEPS";
const SEED_ENV: &str = "REIMAGINE_BURN_REAL_SDXL_SEED";
const DEVICE_ENV: &str = "REIMAGINE_BURN_REAL_SDXL_DEVICE";
const PROMPT_ENV: &str = "REIMAGINE_BURN_REAL_SDXL_PROMPT";
const NEGATIVE_PROMPT_ENV: &str = "REIMAGINE_BURN_REAL_SDXL_NEGATIVE_PROMPT";
const WIDTH_ENV: &str = "REIMAGINE_BURN_REAL_SDXL_WIDTH";
const HEIGHT_ENV: &str = "REIMAGINE_BURN_REAL_SDXL_HEIGHT";
const STOP_AFTER_ENV: &str = "REIMAGINE_BURN_REAL_SDXL_STOP_AFTER";
const CONVERTER_VERSION_MARKER: &str = "burn-sdxl-package-15h-v1";
const RUN_ID: &str = "run-burn-real-sdxl-smoke";
const WORKFLOW_ID: &str = "wf-burn-real-sdxl-smoke";
const DEFAULT_LATENT_SIZE: u32 = 1024;
const LATENT_SCALE: u32 = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
enum RealSdxlSmokeConfig {
    Skipped {
        reason: String,
    },
    Enabled {
        package_root: PathBuf,
        package_origin: RealSdxlPackageOrigin,
        model_id: String,
        steps: u32,
        seed: u64,
        width: u32,
        height: u32,
        stop_after: Option<RealSdxlSmokeStage>,
        device_label: Option<String>,
        prompt: String,
        negative_prompt: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RealSdxlPackageOrigin {
    ExistingPackage,
    DiffusersSplitSource {
        source_root: PathBuf,
        converted_models_root: PathBuf,
    },
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

#[derive(serde::Deserialize)]
struct ConversionReportView {
    package: Option<PackageReportView>,
}

#[derive(serde::Deserialize)]
struct PackageReportView {
    converter_version: String,
}

fn parse_positive_u32(value: &str) -> Option<u32> {
    value.parse::<u32>().ok().filter(|n| *n > 0)
}

fn parse_latent_dim(value: &str, axis: &str) -> Result<u32, String> {
    let dim = parse_positive_u32(value).ok_or_else(|| {
        format!("{axis} must be a positive integer multiple of {LATENT_SCALE}, got `{value}`")
    })?;
    if !dim.is_multiple_of(LATENT_SCALE) {
        return Err(format!(
            "{axis} must be a multiple of {LATENT_SCALE}, got {dim}"
        ));
    }
    Ok(dim)
}

fn parse_stop_after(value: &str) -> Result<RealSdxlSmokeStage, String> {
    match value.trim() {
        "package_validation" => Ok(RealSdxlSmokeStage::PackageValidation),
        "load_bundle" => Ok(RealSdxlSmokeStage::LoadBundle),
        "text_encode" => Ok(RealSdxlSmokeStage::TextEncode),
        "latent_create" => Ok(RealSdxlSmokeStage::LatentCreate),
        "sampler" => Ok(RealSdxlSmokeStage::Sampler),
        "vae_decode" => Ok(RealSdxlSmokeStage::VaeDecode),
        "preview" => Ok(RealSdxlSmokeStage::Preview),
        "save" => Ok(RealSdxlSmokeStage::Save),
        other => Err(format!(
            "unknown {STOP_AFTER_ENV}={other}; expected package_validation|load_bundle|text_encode|latent_create|sampler|vae_decode|preview|save"
        )),
    }
}

impl RealSdxlSmokeConfig {
    fn from_env_getter(get: impl Fn(&str) -> Option<String>) -> Self {
        let width = match get(WIDTH_ENV).filter(|value| !value.trim().is_empty()) {
            Some(raw) => match parse_latent_dim(&raw, "width") {
                Ok(width) => width,
                Err(reason) => return Self::Skipped { reason },
            },
            None => DEFAULT_LATENT_SIZE,
        };
        let height = match get(HEIGHT_ENV).filter(|value| !value.trim().is_empty()) {
            Some(raw) => match parse_latent_dim(&raw, "height") {
                Ok(height) => height,
                Err(reason) => return Self::Skipped { reason },
            },
            None => DEFAULT_LATENT_SIZE,
        };
        let stop_after = match get(STOP_AFTER_ENV).filter(|value| !value.trim().is_empty()) {
            Some(raw) => match parse_stop_after(&raw) {
                Ok(stage) => Some(stage),
                Err(reason) => return Self::Skipped { reason },
            },
            None => None,
        };

        match (
            get(PACKAGE_ROOT_ENV).filter(|value| !value.trim().is_empty()),
            get(SPLIT_SOURCE_ROOT_ENV).filter(|value| !value.trim().is_empty()),
        ) {
            (Some(package_root), _) => Self::Enabled {
                package_root: PathBuf::from(package_root),
                package_origin: RealSdxlPackageOrigin::ExistingPackage,
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
                width,
                height,
                stop_after,
                device_label: get(DEVICE_ENV).filter(|value| !value.trim().is_empty()),
                prompt: get(PROMPT_ENV)
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or_else(|| "small bright city at sunrise".to_owned()),
                negative_prompt: get(NEGATIVE_PROMPT_ENV)
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or_else(|| "low quality blur".to_owned()),
            },
            (None, Some(split_source_root)) => {
                let source_root = PathBuf::from(split_source_root);
                let converted_models_root = infer_converted_models_root(&source_root);
                Self::Enabled {
                    package_root: converted_models_root
                        .join("burn")
                        .join("burn-real-sdxl-smoke")
                        .join("pending"),
                    package_origin: RealSdxlPackageOrigin::DiffusersSplitSource {
                        source_root,
                        converted_models_root,
                    },
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
                    width,
                    height,
                    stop_after,
                    device_label: get(DEVICE_ENV).filter(|value| !value.trim().is_empty()),
                    prompt: get(PROMPT_ENV)
                        .filter(|value| !value.trim().is_empty())
                        .unwrap_or_else(|| "small bright city at sunrise".to_owned()),
                    negative_prompt: get(NEGATIVE_PROMPT_ENV)
                        .filter(|value| !value.trim().is_empty())
                        .unwrap_or_else(|| "low quality blur".to_owned()),
                }
            }
            (None, None) => Self::Skipped {
                reason: format!(
                    "set {PACKAGE_ROOT_ENV} to a converted Burn SDXL package root or {SPLIT_SOURCE_ROOT_ENV} to a local diffusers-style split source"
                ),
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
            reason: format!(
                "set {PACKAGE_ROOT_ENV} to a converted Burn SDXL package root or {SPLIT_SOURCE_ROOT_ENV} to a local diffusers-style split source"
            )
        }
    );
}

#[test]
fn real_sdxl_smoke_can_bootstrap_package_from_local_split_source() {
    let source_root = PathBuf::from("/workspace/models/converted/sd_xl_base_1.0/size-6938078334");
    let config = RealSdxlSmokeConfig::from_env_getter(|key| match key {
        SPLIT_SOURCE_ROOT_ENV => Some(source_root.display().to_string()),
        _ => None,
    });

    assert_eq!(
        config,
        RealSdxlSmokeConfig::Enabled {
            package_root: PathBuf::from(
                "/workspace/models/converted/burn/burn-real-sdxl-smoke/pending"
            ),
            package_origin: RealSdxlPackageOrigin::DiffusersSplitSource {
                source_root,
                converted_models_root: PathBuf::from("/workspace/models/converted"),
            },
            model_id: "burn-real-sdxl-smoke".to_owned(),
            steps: 1,
            seed: 1234,
            width: DEFAULT_LATENT_SIZE,
            height: DEFAULT_LATENT_SIZE,
            stop_after: None,
            device_label: None,
            prompt: "small bright city at sunrise".to_owned(),
            negative_prompt: "low quality blur".to_owned(),
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
        package_origin,
        ..
    } = config
    else {
        panic!("expected enabled smoke config");
    };
    assert_eq!(package_origin, RealSdxlPackageOrigin::ExistingPackage);

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
        err.contains("expected converter version `burn-sdxl-package-15h-v1`"),
        "{err}"
    );
}

#[tokio::test]
async fn real_sdxl_component_package_runs_public_burn_capability_chain_when_enabled() {
    let config = RealSdxlSmokeConfig::from_env_getter(|key| std::env::var(key).ok());
    let RealSdxlSmokeConfig::Enabled {
        package_root: configured_package_root,
        package_origin,
        model_id,
        steps,
        seed,
        width,
        height,
        stop_after,
        device_label,
        prompt,
        negative_prompt,
    } = config
    else {
        eprintln!(
            "skipping real SDXL Burn smoke: set {PACKAGE_ROOT_ENV} to a converted Burn SDXL package root or {SPLIT_SOURCE_ROOT_ENV} to a local diffusers-style split source"
        );
        return;
    };

    let package_root = match package_origin {
        RealSdxlPackageOrigin::ExistingPackage => configured_package_root,
        RealSdxlPackageOrigin::DiffusersSplitSource {
            source_root,
            converted_models_root,
        } => stage_context(
            RealSdxlSmokeStage::PackageValidation,
            package_split_source(&source_root, &converted_models_root, &model_id),
        )
        .expect("package split source"),
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
    if stop_after == Some(RealSdxlSmokeStage::PackageValidation) {
        eprintln!("real SDXL Burn smoke stopped after package_validation");
        return;
    }

    let model = resolved_model_from_package(&package_root, &model_id);
    eprintln!(
        "real SDXL Burn smoke: package_root={}, backend_device={}, converter_marker={}, size={}x{}, steps={}, seed={}, stop_after={}, sampler=euler, scheduler=normal",
        package_root.display(),
        backend.config().device_label(),
        CONVERTER_VERSION_MARKER,
        width,
        height,
        steps,
        seed,
        stop_after.map(RealSdxlSmokeStage::as_str).unwrap_or("none")
    );

    let loaded = stage_context(
        RealSdxlSmokeStage::LoadBundle,
        backend.load_bundle(load_request(model)).await,
    )
    .expect("load_bundle");
    eprintln!(
        "real SDXL Burn smoke load_bundle ok: model={}, clip={}, vae={}",
        loaded.model().payload_key().as_str(),
        loaded.clip().payload_key().as_str(),
        loaded.vae().payload_key().as_str()
    );
    if stop_after == Some(RealSdxlSmokeStage::LoadBundle) {
        eprintln!("real SDXL Burn smoke stopped after load_bundle");
        return;
    }

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
    if stop_after == Some(RealSdxlSmokeStage::TextEncode) {
        eprintln!("real SDXL Burn smoke stopped after text_encode");
        return;
    }

    let latent = stage_context(
        RealSdxlSmokeStage::LatentCreate,
        backend
            .create_empty_latent(CreateEmptyLatentRequest::new(
                width,
                height,
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
    if stop_after == Some(RealSdxlSmokeStage::LatentCreate) {
        eprintln!("real SDXL Burn smoke stopped after latent_create");
        return;
    }

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
    if stop_after == Some(RealSdxlSmokeStage::Sampler) {
        eprintln!("real SDXL Burn smoke stopped after sampler");
        return;
    }

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
    if stop_after == Some(RealSdxlSmokeStage::VaeDecode) {
        eprintln!("real SDXL Burn smoke stopped after vae_decode");
        return;
    }

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
    if stop_after == Some(RealSdxlSmokeStage::Preview) {
        assert_png_artifact(output.path(), preview.as_str());
        eprintln!(
            "real SDXL Burn smoke stopped after preview: {}",
            preview.as_str()
        );
        return;
    }

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

fn package_split_source(
    source_root: &std::path::Path,
    converted_models_root: &std::path::Path,
    model_id: &str,
) -> Result<PathBuf, String> {
    validate_split_source_root(source_root)?;
    let result = package_diffusers_style_split_sdxl_source(&BurnSdxlDiffusersSplitPackageRequest {
        source_root: source_root.to_path_buf(),
        source_model_id: model_id.to_owned(),
        source_fingerprint: None,
        converted_models_root: converted_models_root.to_path_buf(),
        overwrite: false,
    })
    .map_err(|err| err.to_string())?;
    validate_package_root(&result.package_root)?;
    eprintln!(
        "real SDXL Burn smoke packaged split source: source_root={}, package_root={}, report={}, reused_existing={}",
        source_root.display(),
        result.package_root.display(),
        result.report_path.display(),
        result.reused_existing
    );
    Ok(result.package_root)
}

fn validate_split_source_root(source_root: &std::path::Path) -> Result<(), String> {
    if !source_root.is_dir() {
        return Err(format!(
            "split source root `{}` is not a directory",
            source_root.display()
        ));
    }
    for role_dir in ["unet", "vae", "text_encoder", "text_encoder_2"] {
        let component = source_root.join(role_dir).join("model.safetensors");
        if !component.is_file() {
            return Err(format!(
                "missing split source component `{}`",
                component.display()
            ));
        }
    }
    Ok(())
}

fn infer_converted_models_root(source_root: &std::path::Path) -> PathBuf {
    source_root
        .parent()
        .and_then(std::path::Path::parent)
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| source_root.join("converted"))
}

fn assert_png_artifact(output_root: &std::path::Path, artifact: &str) {
    assert!(
        artifact.ends_with(".png"),
        "artifact should be png: {artifact}"
    );
    // Runtime artifact refs are usually `output/<filename>` relative to workspace;
    // smoke output_root is already the backend output_dir, so strip that prefix.
    let relative = artifact
        .strip_prefix("output/")
        .or_else(|| artifact.strip_prefix("output\\"))
        .unwrap_or(artifact);
    let path = if std::path::Path::new(artifact).is_absolute() {
        std::path::PathBuf::from(artifact)
    } else {
        output_root.join(relative)
    };
    assert!(path.is_file(), "artifact file exists: {}", path.display());
    let bytes = std::fs::read(&path).expect("read artifact");
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
