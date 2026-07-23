use reimagine_core::diagnostic::DiagnosticSeverity;
use reimagine_core::model::{ModelId, ModelRole, ModelSeries, ModelVariant};
use reimagine_model_manager::{
    Fingerprint, IdPolicy, IdResolution, ModelDescriptor, ModelFormat, ModelRootId, ModelSource,
    ModelSourceStatus,
};

fn test_fingerprint(value: &str) -> Fingerprint {
    Fingerprint::sha256(value)
}

fn descriptor(
    id: &str,
    series: &str,
    variant: &str,
    roles: Vec<ModelRole>,
    source: ModelSource,
    format: ModelFormat,
) -> ModelDescriptor {
    ModelDescriptor::new(
        ModelId::new(id),
        ModelSeries::new(series),
        ModelVariant::new(variant),
        roles,
        source,
        format,
    )
}

// --- Manual ID conflict tests ---

#[test]
fn manual_id_no_conflict() {
    let existing = vec![descriptor(
        "my-model",
        "stable_diffusion",
        "sdxl",
        vec![ModelRole::CheckpointBundle],
        ModelSource::relative(ModelRootId::new("base"), "checkpoints/model.safetensors"),
        ModelFormat::Safetensors,
    )];
    let policy = IdPolicy::new(&existing);
    let report = policy.validate_manual_id("other-model");
    assert!(report.is_empty());
}

#[test]
fn manual_id_conflict_rejected() {
    let existing = vec![descriptor(
        "my-model",
        "stable_diffusion",
        "sdxl",
        vec![ModelRole::CheckpointBundle],
        ModelSource::relative(ModelRootId::new("base"), "checkpoints/model.safetensors"),
        ModelFormat::Safetensors,
    )];
    let policy = IdPolicy::new(&existing);
    let report = policy.validate_manual_id("my-model");
    assert_eq!(report.diagnostics().len(), 1);
    assert_eq!(
        report.diagnostics()[0].code().as_str(),
        "MODEL_MANAGER/MANUAL_ID_CONFLICT"
    );
    assert_eq!(
        report.diagnostics()[0].severity(),
        DiagnosticSeverity::Error
    );
}

// --- Auto ID deterministic generation tests ---

#[test]
fn auto_id_deterministic_for_same_input() {
    let policy = IdPolicy::new(&[]);
    let source = ModelSource::relative(
        ModelRootId::new("base"),
        "checkpoints/sdxl_base_1.0.safetensors",
    );
    let id1 = policy.generate_auto_id(
        &ModelSeries::new("stable_diffusion"),
        &ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        &source,
    );
    let id2 = policy.generate_auto_id(
        &ModelSeries::new("stable_diffusion"),
        &ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        &source,
    );
    assert_eq!(id1, id2);
}

#[test]
fn auto_id_contains_series_variant_role_normalized_filename() {
    let policy = IdPolicy::new(&[]);
    let source = ModelSource::relative(
        ModelRootId::new("base"),
        "checkpoints/sdxl_base_1.0.safetensors",
    );
    let id = policy.generate_auto_id(
        &ModelSeries::new("stable_diffusion"),
        &ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        &source,
    );
    // Should contain the expected parts separated by hyphens.
    assert!(id.contains("stable_diffusion"), "id={id}");
    assert!(id.contains("sdxl"), "id={id}");
    assert!(id.contains("CheckpointBundle"), "id={id}");
    assert!(id.contains("sdxl_base_1_0"), "id={id}");
    // Should end with a short hash segment.
    let last_segment = id.rsplit('-').next().unwrap();
    assert_eq!(
        last_segment.len(),
        8,
        "hash segment should be 8 hex chars: {id}"
    );
    assert!(
        last_segment.chars().all(|c| c.is_ascii_hexdigit()),
        "hash should be hex: {id}"
    );
}

#[test]
fn auto_id_normalizes_filename() {
    let policy = IdPolicy::new(&[]);
    let source = ModelSource::relative(
        ModelRootId::new("base"),
        "models/My Cool Model (v2).safetensors",
    );
    let id = policy.generate_auto_id(
        &ModelSeries::new("test"),
        &ModelVariant::new("v1"),
        ModelRole::Lora,
        &source,
    );
    // Filename stem "My Cool Model (v2)" normalizes to "my_cool_model_v2".
    assert!(id.contains("my_cool_model_v2"), "id={id}");
}

#[test]
fn auto_id_different_inputs_produce_different_ids() {
    let policy = IdPolicy::new(&[]);
    let source_a =
        ModelSource::relative(ModelRootId::new("base"), "checkpoints/model_a.safetensors");
    let source_b =
        ModelSource::relative(ModelRootId::new("base"), "checkpoints/model_b.safetensors");
    let id_a = policy.generate_auto_id(
        &ModelSeries::new("stable_diffusion"),
        &ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        &source_a,
    );
    let id_b = policy.generate_auto_id(
        &ModelSeries::new("stable_diffusion"),
        &ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        &source_b,
    );
    assert_ne!(id_a, id_b);
}

// --- Auto ID collision handling tests ---

#[test]
fn auto_id_collision_same_fingerprint_resolves_to_same() {
    let fp = test_fingerprint("abc123");
    let source = ModelSource::relative(
        ModelRootId::new("base"),
        "checkpoints/sdxl_model.safetensors",
    );
    let generated_id = IdPolicy::new(&[]).generate_auto_id(
        &ModelSeries::new("stable_diffusion"),
        &ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        &source,
    );
    let existing = vec![
        descriptor(
            &generated_id,
            "stable_diffusion",
            "sdxl",
            vec![ModelRole::CheckpointBundle],
            source.clone(),
            ModelFormat::Safetensors,
        )
        .with_fingerprint(fp.clone())
        .with_source_status(ModelSourceStatus::Available),
    ];
    let policy = IdPolicy::new(&existing);
    // Same fingerprint and source resolves as the same model.
    let result = policy.generate_auto_id_with_resolution(
        &ModelSeries::new("stable_diffusion"),
        &ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        &source,
        Some(&fp),
    );
    assert_eq!(result.id().as_str(), generated_id);
    assert_eq!(result.resolution(), IdResolution::SameIdentity);
    assert!(result.report().is_empty());
}

#[test]
fn auto_id_collision_different_fingerprint_suffixes_and_diagnostic() {
    let source = ModelSource::relative(
        ModelRootId::new("base"),
        "checkpoints/sdxl_model.safetensors",
    );
    let generated_id = IdPolicy::new(&[]).generate_auto_id(
        &ModelSeries::new("stable_diffusion"),
        &ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        &source,
    );
    let existing = vec![
        descriptor(
            &generated_id,
            "stable_diffusion",
            "sdxl",
            vec![ModelRole::CheckpointBundle],
            source.clone(),
            ModelFormat::Safetensors,
        )
        .with_fingerprint(test_fingerprint("abc123"))
        .with_source_status(ModelSourceStatus::Available),
    ];

    let policy = IdPolicy::new(&existing);
    let result = policy.generate_auto_id_with_resolution(
        &ModelSeries::new("stable_diffusion"),
        &ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        &source,
        Some(&test_fingerprint("different_hash")),
    );
    // Collision resolved by suffixing longer hash.
    assert_ne!(
        result.id().as_str(),
        generated_id,
        "id should be different after suffixing"
    );
    assert_eq!(result.resolution(), IdResolution::SuffixAppended);
    // Should still contain the core parts.
    assert!(result.id().as_str().contains("stable_diffusion"));
    assert!(result.id().as_str().contains("sdxl"));
    assert!(result.id().as_str().contains("CheckpointBundle"));

    let report = result.report();
    assert_eq!(report.diagnostics().len(), 1);
    assert_eq!(
        report.diagnostics()[0].code().as_str(),
        "MODEL_MANAGER/AUTO_ID_COLLISION_RESOLVED"
    );
}

#[test]
fn auto_id_collision_avoids_taken_suffixed_id() {
    let source = ModelSource::relative(
        ModelRootId::new("base"),
        "checkpoints/sdxl_model.safetensors",
    );
    let base_id = IdPolicy::new(&[]).generate_auto_id(
        &ModelSeries::new("stable_diffusion"),
        &ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        &source,
    );

    let first_collision = descriptor(
        &base_id,
        "stable_diffusion",
        "sdxl",
        vec![ModelRole::CheckpointBundle],
        source.clone(),
        ModelFormat::Safetensors,
    )
    .with_fingerprint(test_fingerprint("first"))
    .with_source_status(ModelSourceStatus::Available);

    let first_result = IdPolicy::new(std::slice::from_ref(&first_collision))
        .generate_auto_id_with_resolution(
            &ModelSeries::new("stable_diffusion"),
            &ModelVariant::new("sdxl"),
            ModelRole::CheckpointBundle,
            &source,
            Some(&test_fingerprint("second")),
        );

    let taken_suffixed_id = first_result.id().as_str().to_owned();
    let existing = vec![
        first_collision,
        descriptor(
            &taken_suffixed_id,
            "stable_diffusion",
            "sdxl",
            vec![ModelRole::CheckpointBundle],
            ModelSource::relative(ModelRootId::new("base"), "other/sdxl_model.safetensors"),
            ModelFormat::Safetensors,
        )
        .with_fingerprint(test_fingerprint("third"))
        .with_source_status(ModelSourceStatus::Available),
    ];

    let result = IdPolicy::new(&existing).generate_auto_id_with_resolution(
        &ModelSeries::new("stable_diffusion"),
        &ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        &source,
        Some(&test_fingerprint("second")),
    );

    assert_eq!(result.resolution(), IdResolution::SuffixAppended);
    assert_ne!(result.id().as_str(), base_id);
    assert_ne!(result.id().as_str(), taken_suffixed_id);
    assert!(result.id().as_str().ends_with("-2"), "id={}", result.id());
}

#[test]
fn auto_id_no_collision_when_id_unused() {
    let policy = IdPolicy::new(&[]);
    let source = ModelSource::relative(
        ModelRootId::new("base"),
        "checkpoints/sdxl_model.safetensors",
    );
    let result = policy.generate_auto_id_with_resolution(
        &ModelSeries::new("stable_diffusion"),
        &ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        &source,
        Some(&test_fingerprint("abc")),
    );
    assert!(result.id().as_str().contains("stable_diffusion"));
    assert_eq!(result.resolution(), IdResolution::NoConflict);
    assert!(result.report().is_empty());
}

#[test]
fn same_fingerprint_different_source_not_same_identity() {
    let fp = test_fingerprint("abc123");
    let existing_source = ModelSource::relative(
        ModelRootId::new("base"),
        "checkpoints/sdxl_model.safetensors",
    );
    let generated_id = IdPolicy::new(&[]).generate_auto_id(
        &ModelSeries::new("stable_diffusion"),
        &ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        &existing_source,
    );
    let existing = vec![
        descriptor(
            &generated_id,
            "stable_diffusion",
            "sdxl",
            vec![ModelRole::CheckpointBundle],
            existing_source,
            ModelFormat::Safetensors,
        )
        .with_fingerprint(fp.clone())
        .with_source_status(ModelSourceStatus::Available),
    ];

    let policy = IdPolicy::new(&existing);
    // Different source path should not be treated as the same identity.
    let source = ModelSource::relative(
        ModelRootId::new("base"),
        "loras/sdxl_model_copy.safetensors",
    );
    let result = policy.generate_auto_id_with_resolution(
        &ModelSeries::new("stable_diffusion"),
        &ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        &source,
        Some(&fp),
    );
    assert_eq!(result.resolution(), IdResolution::NoConflict);
}

#[test]
fn same_fingerprint_same_relative_path_different_root_not_same_identity() {
    let fp = test_fingerprint("abc123");
    let source = ModelSource::relative(ModelRootId::new("base"), "checkpoints/model.safetensors");
    let generated_id = IdPolicy::new(&[]).generate_auto_id(
        &ModelSeries::new("stable_diffusion"),
        &ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        &source,
    );
    let existing = vec![
        descriptor(
            &generated_id,
            "stable_diffusion",
            "sdxl",
            vec![ModelRole::CheckpointBundle],
            source,
            ModelFormat::Safetensors,
        )
        .with_fingerprint(fp.clone())
        .with_source_status(ModelSourceStatus::Available),
    ];

    let result = IdPolicy::new(&existing).generate_auto_id_with_resolution(
        &ModelSeries::new("stable_diffusion"),
        &ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        &ModelSource::relative(
            ModelRootId::new("external"),
            "checkpoints/model.safetensors",
        ),
        Some(&fp),
    );

    assert_eq!(result.resolution(), IdResolution::SuffixAppended);
}
