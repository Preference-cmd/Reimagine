//! Library target for the Burn inference worker.
//!
//! Re-exports the `probe` module so integration tests can
//! verify device profile and identity construction without
//! spawning a subprocess. The binary entry point lives in
//! `main.rs` and is linked separately.

pub mod probe;

#[cfg(test)]
mod tests {
    use super::probe;
    use reimagine_inference_burn::{BurnBackend, BurnBackendConfig};

    #[test]
    fn probe_reports_burn_backend_kind() {
        let backend = BurnBackend::new(BurnBackendConfig::new("/tmp/models", "/tmp/output"))
            .expect("failed to create BurnBackend");
        let (identity, profile) = probe::build(&backend);
        assert_eq!(identity.backend_kind, "burn");
        assert!(!identity.worker_version.is_empty());
        assert!(!identity.target.is_empty());
        assert!(!identity.incarnation_id.0.is_empty());
        assert!(!profile.instances.is_empty());
    }

    #[test]
    fn probe_device_label_matches_compiled_feature() {
        let backend = BurnBackend::new(BurnBackendConfig::new("/tmp/models", "/tmp/output"))
            .expect("failed to create BurnBackend");
        let (_identity, profile) = probe::build(&backend);
        let instance = &profile.instances[0];
        #[cfg(feature = "wgpu")]
        assert!(
            instance.device_label.starts_with("wgpu:"),
            "wgpu device label should start with 'wgpu:', got {}",
            instance.device_label
        );
        #[cfg(all(not(feature = "wgpu"), feature = "flex"))]
        assert_eq!(instance.device_label, "flex:cpu");
    }

    #[test]
    fn probe_advertises_expected_capabilities() {
        let backend = BurnBackend::new(BurnBackendConfig::new("/tmp/models", "/tmp/output"))
            .expect("failed to create BurnBackend");
        let (_identity, profile) = probe::build(&backend);
        let caps = &profile.instances[0].capabilities;
        let expected = [
            "model.load_bundle",
            "text.encode",
            "latent.create_empty",
            "diffusion.sample",
            "latent.decode",
            "image.save",
            "image.preview",
        ];
        for cap in &expected {
            assert!(caps.contains(&cap.to_string()), "missing capability: {cap}");
        }
        assert_eq!(caps.len(), expected.len());
    }
}
