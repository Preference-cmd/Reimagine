use reimagine_model_manager::ModelDescriptor;
use serde::{Deserialize, Serialize};

/// Host-neutral projection of a model suitable for both Tauri and Axum.
///
/// This shape strips absolute filesystem paths and backend‑internal types
/// so frontends never see arbitrary paths or model‑manager internals.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfoDto {
    /// Canonical model identifier (e.g. `sd_xl_base_1_0`).
    pub id: String,
    /// Human‑readable display label.
    pub display_name: String,
    /// Model series (e.g. `stable-diffusion-xl`).
    pub model_series: String,
    /// Model variant (e.g. `base`).
    pub variant: String,
    /// Roles this model can fulfil (e.g. `["checkpoint-bundle", "diffusion-model"]`).
    pub roles: Vec<String>,
    /// File format (e.g. `safetensors`).
    pub format: String,
    /// Source availability status (e.g. `available`, `missing`, `unverified`).
    pub source_status: String,
    /// File size in bytes, if known.
    pub size_bytes: Option<u64>,
}

impl From<ModelDescriptor> for ModelInfoDto {
    fn from(descriptor: ModelDescriptor) -> Self {
        let id = descriptor.id().as_str().to_owned();
        let series = descriptor.model_series().as_str().to_owned();
        let variant = descriptor.variant().as_str().to_owned();

        // Derive a human‑readable display name from series + variant.
        let display_name = format_display_name(&series, &variant);

        Self {
            id,
            display_name,
            model_series: series,
            variant,
            roles: descriptor
                .roles()
                .iter()
                .map(|r| format_role(r))
                .collect(),
            format: format!("{:?}", descriptor.format()).to_ascii_lowercase(),
            source_status: format_status(descriptor.source_status()),
            size_bytes: descriptor.size_bytes(),
        }
    }
}

fn format_display_name(series: &str, variant: &str) -> String {
    let series = series.replace(['-', '_'], " ");
    let variant = variant.replace(['-', '_'], " ");
    let series = capitalize_words(&series);
    let variant = capitalize_words(&variant);
    format!("{series} {variant}")
}

fn capitalize_words(s: &str) -> String {
    s.split_whitespace()
        .map(|w| {
            if w.is_empty() {
                String::new()
            } else {
                let mut chars = w.chars();
                let first = chars.next().unwrap().to_uppercase().to_string();
                let rest: String = chars.collect();
                first + rest.as_str()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_role(role: &reimagine_core::model::ModelRole) -> String {
    use reimagine_core::model::ModelRole;
    match role {
        ModelRole::CheckpointBundle => "checkpoint-bundle".to_owned(),
        ModelRole::DiffusionModel => "diffusion-model".to_owned(),
        ModelRole::TextEncoder => "text-encoder".to_owned(),
        ModelRole::Vae => "vae".to_owned(),
        ModelRole::Scheduler => "scheduler".to_owned(),
        ModelRole::Lora => "lora".to_owned(),
        ModelRole::ControlNet => "controlnet".to_owned(),
        ModelRole::Upscaler => "upscaler".to_owned(),
    }
}

fn format_status(status: reimagine_model_manager::ModelSourceStatus) -> String {
    use reimagine_model_manager::ModelSourceStatus;
    match status {
        ModelSourceStatus::Available => "available".to_owned(),
        ModelSourceStatus::Missing => "missing".to_owned(),
        ModelSourceStatus::Stale => "stale".to_owned(),
        ModelSourceStatus::Unverified => "unverified".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_core::model::{
        ModelId, ModelRole, ModelSeries, ModelVariant,
    };
    use reimagine_model_manager::{
        ModelDescriptor, ModelFormat, ModelSource,
    };

    fn make_descriptor(id: &str, series: &str, variant: &str, roles: Vec<ModelRole>) -> ModelDescriptor {
        ModelDescriptor::new(
            ModelId::new(id),
            ModelSeries::new(series),
            ModelVariant::new(variant),
            roles,
            ModelSource::relative(
                reimagine_model_manager::ModelRootId::new("base"),
                format!("models/{id}.safetensors"),
            ),
            ModelFormat::Safetensors,
        )
        .with_size_bytes(7_456_123_456)
        .with_source_status(reimagine_model_manager::ModelSourceStatus::Available)
    }

    #[test]
    fn projects_sdxl_base_descriptor() {
        let descriptor = make_descriptor(
            "sd_xl_base_1_0",
            "stable-diffusion-xl",
            "base",
            vec![ModelRole::CheckpointBundle, ModelRole::DiffusionModel],
        );
        let dto: ModelInfoDto = descriptor.into();

        assert_eq!(dto.id, "sd_xl_base_1_0");
        assert_eq!(dto.display_name, "Stable Diffusion Xl Base");
        assert_eq!(dto.model_series, "stable-diffusion-xl");
        assert_eq!(dto.variant, "base");
        assert_eq!(dto.roles, vec!["checkpoint-bundle", "diffusion-model"]);
        assert_eq!(dto.format, "safetensors");
        assert_eq!(dto.source_status, "available");
        assert_eq!(dto.size_bytes, Some(7_456_123_456));
    }

    #[test]
    fn projects_dreamshaper_descriptor() {
        let descriptor = make_descriptor(
            "dreamshaper_8",
            "stable-diffusion-1.5",
            "dreamshaper",
            vec![ModelRole::CheckpointBundle],
        );
        let dto: ModelInfoDto = descriptor.into();

        assert_eq!(dto.id, "dreamshaper_8");
        assert_eq!(dto.display_name, "Stable Diffusion 1.5 Dreamshaper");
        assert_eq!(dto.model_series, "stable-diffusion-1.5");
        assert_eq!(dto.variant, "dreamshaper");
        assert_eq!(dto.roles, vec!["checkpoint-bundle"]);
        assert_eq!(dto.source_status, "available");
    }

    #[test]
    fn handles_converted_model_with_components() {
        let descriptor = ModelDescriptor::new(
            ModelId::new("my_converted_model"),
            ModelSeries::new("stable-diffusion-xl"),
            ModelVariant::new("base"),
            vec![ModelRole::DiffusionModel, ModelRole::TextEncoder, ModelRole::Vae],
            ModelSource::relative(
                reimagine_model_manager::ModelRootId::new("base"),
                "models/converted/src_id/fp/sdxl",
            ),
            ModelFormat::Safetensors,
        )
        .with_size_bytes(0)
        .with_source_status(reimagine_model_manager::ModelSourceStatus::Available);

        let dto: ModelInfoDto = descriptor.into();

        assert_eq!(dto.id, "my_converted_model");
        assert_eq!(dto.display_name, "Stable Diffusion Xl Base");
        assert_eq!(dto.size_bytes, Some(0));
        assert!(dto.roles.contains(&"diffusion-model".to_owned()));
        assert!(dto.roles.contains(&"text-encoder".to_owned()));
        assert!(dto.roles.contains(&"vae".to_owned()));
    }

    #[test]
    fn missing_size_is_none() {
        let descriptor = make_descriptor(
            "no_size_model",
            "test-series",
            "test-variant",
            vec![ModelRole::CheckpointBundle],
        );
        // Override to remove size
        let minimal = ModelDescriptor::new(
            descriptor.id().clone(),
            descriptor.model_series().clone(),
            descriptor.variant().clone(),
            descriptor.roles().to_vec(),
            ModelSource::relative(
                reimagine_model_manager::ModelRootId::new("base"),
                "nowhere.safetensors",
            ),
            ModelFormat::Safetensors,
        );
        let dto: ModelInfoDto = minimal.into();
        assert_eq!(dto.display_name, "Test Series Test Variant");
        assert_eq!(dto.size_bytes, None);
    }

    #[test]
    fn formats_all_source_statuses() {
        use reimagine_model_manager::ModelSourceStatus;

        for (status, expected) in [
            (ModelSourceStatus::Available, "available"),
            (ModelSourceStatus::Missing, "missing"),
            (ModelSourceStatus::Stale, "stale"),
            (ModelSourceStatus::Unverified, "unverified"),
        ] {
            let descriptor = ModelDescriptor::new(
                ModelId::new("test"),
                ModelSeries::new("s"),
                ModelVariant::new("v"),
                vec![],
                ModelSource::relative(
                    reimagine_model_manager::ModelRootId::new("base"),
                    "test.safetensors",
                ),
                ModelFormat::Safetensors,
            )
            .with_source_status(status);
            let dto: ModelInfoDto = descriptor.into();
            assert_eq!(dto.source_status, expected, "mismatch for {status:?}");
        }
    }
}
