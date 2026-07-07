use std::path::Path;

use crate::config::ModelAcquisitionConfig;

/// Build an `HFClient` from the model-acquisition config.
///
/// Uses explicit token from config — uses `HFClient::builder()` (not `HFClient::new()`)
/// to avoid implicit env-var-based token discovery.
pub fn build_hf_client(config: &ModelAcquisitionConfig) -> hf_hub::HFClient {
    let mut builder = hf_hub::HFClient::builder().endpoint(config.huggingface.endpoint.clone());

    if let Some(ref token) = config.huggingface.token {
        builder = builder.token(token.clone());
    }

    if let Some(ref cache_dir) = config.huggingface.cache_dir {
        builder = builder.cache_dir(cache_dir.clone());
    }

    if !config.huggingface.cache_enabled {
        builder = builder.cache_enabled(false);
    }

    builder
        .build()
        .expect("failed to build HFClient — this should not panic in normal use")
}

/// Helper to get the cache directory from config, or the default hf-hub cache.
pub fn hf_cache_dir(config: &ModelAcquisitionConfig) -> Option<&Path> {
    config.huggingface.cache_dir.as_deref()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelAcquisitionConfig;

    #[test]
    fn test_build_hf_client_default() {
        let config = ModelAcquisitionConfig::default();
        let _client = build_hf_client(&config);
        // Just verify it doesn't panic — the client is lazily initialized.
    }

    #[test]
    fn test_build_hf_client_with_token() {
        let mut config = ModelAcquisitionConfig::default();
        config.huggingface.token = Some("hf_test_token".to_owned());
        let _client = build_hf_client(&config);
    }
}
