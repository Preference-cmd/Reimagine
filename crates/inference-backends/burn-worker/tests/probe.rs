//! Profile and device-selection conformance tests for the Burn worker.
//!
//! Verifies that the probe module reports the correct identity and
//! capability profile based on the compiled feature (wgpu vs flex).

use reimagine_inference_burn::{BurnBackend, BurnBackendConfig};
use reimagine_inference_burn_worker::probe;

/// Test that `probe::build` returns a non-empty identity with the
/// expected backend kind and a profile containing at least one instance.
#[test]
fn probe_reports_burn_backend_kind() {
    let backend = BurnBackend::new(BurnBackendConfig::new("/tmp/models", "/tmp/output"))
        .expect("failed to create BurnBackend for probe test");
    let (identity, profile) = probe::build(&backend);

    assert_eq!(
        identity.backend_kind, "burn",
        "worker backend kind must be 'burn'"
    );
    assert!(
        !identity.worker_version.is_empty(),
        "worker version must be non-empty"
    );
    assert!(
        !identity.target.is_empty(),
        "target triplet must be non-empty"
    );
    assert!(
        !identity.incarnation_id.0.is_empty(),
        "incarnation id must be non-empty"
    );

    // Profile must have at least one instance
    assert!(
        !profile.instances.is_empty(),
        "profile must contain at least one instance"
    );
}

/// Test that the probe's reported device label matches the compiled
/// feature (wgpu:default under wgpu, flex:cpu under flex).
#[test]
fn probe_device_label_matches_compiled_feature() {
    let backend = BurnBackend::new(BurnBackendConfig::new("/tmp/models", "/tmp/output"))
        .expect("failed to create BurnBackend for probe test");
    let (_identity, profile) = probe::build(&backend);

    let instance = &profile.instances[0];
    #[cfg(feature = "wgpu")]
    assert!(
        instance.device_label.starts_with("wgpu:"),
        "under wgpu feature, device label should start with 'wgpu:', got '{}'",
        instance.device_label
    );
    #[cfg(all(not(feature = "wgpu"), feature = "flex"))]
    assert_eq!(
        instance.device_label, "flex:cpu",
        "under flex feature, device label should be 'flex:cpu'"
    );
}

/// Test that the probe reports a comprehensive capability set.
#[test]
fn probe_capabilities_cover_all_sdxl_operations() {
    let backend = BurnBackend::new(BurnBackendConfig::new("/tmp/models", "/tmp/output"))
        .expect("failed to create BurnBackend for probe test");
    let (_identity, profile) = probe::build(&backend);

    let instance = &profile.instances[0];
    let required_caps = [
        "model.load_bundle",
        "latent.create_empty",
        "text.encode",
        "diffusion.sample",
        "latent.decode",
        "image.save",
        "image.preview",
    ];

    for cap in &required_caps {
        assert!(
            instance.capabilities.iter().any(|c| c == cap),
            "probe should report capability '{}' but it is missing from {:#?}",
            cap,
            instance.capabilities
        );
    }
}
