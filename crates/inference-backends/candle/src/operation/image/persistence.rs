//! Image artifact persistence and output filename/path policy.

use reimagine_core::model::ArtifactRef;
use reimagine_inference::{
    FilenamePrefix, ImagePreviewRequest, ImagePreviewResponse, ImageSaveRequest, ImageSaveResponse,
    InferenceBackend, RuntimeImage,
};

use super::encoding::encode_tensor_to_png;
use crate::backend::CandleBackend;
use crate::error::CandleBackendError;

pub fn execute_image_save(
    request: ImageSaveRequest,
    backend: &CandleBackend,
) -> Result<ImageSaveResponse, CandleBackendError> {
    let prefix = match request.filename_prefix() {
        FilenamePrefix::Default => "reimagine".to_string(),
        FilenamePrefix::Custom(s) => s.clone(),
    };
    let run_id = request.run_id().clone();
    let node_id = request.node_id().clone();
    let response_artifact =
        persist_image(request.into_image(), &prefix, &run_id, &node_id, backend)?;
    Ok(ImageSaveResponse::new(response_artifact))
}

pub fn execute_image_preview(
    request: ImagePreviewRequest,
    backend: &CandleBackend,
) -> Result<ImagePreviewResponse, CandleBackendError> {
    let run_id = request.run_id().clone();
    let node_id = request.node_id().clone();
    let response_artifact =
        persist_image(request.into_image(), "preview", &run_id, &node_id, backend)?;
    Ok(ImagePreviewResponse::new(response_artifact))
}

fn persist_image(
    image_value: RuntimeImage,
    prefix: &str,
    run_id: &reimagine_core::model::RunId,
    node_id: &reimagine_core::model::NodeId,
    backend: &CandleBackend,
) -> Result<ArtifactRef, CandleBackendError> {
    if image_value.payload().backend() != backend.backend_kind() {
        return Err(CandleBackendError::InvalidRequest(format!(
            "image.save `image` handle belongs to backend `{}`, expected `{}`",
            image_value.payload().backend().as_str(),
            backend.backend_kind()
        )));
    }

    let payload_key = image_value.payload().payload_key();
    if !backend.store().contains_payload(payload_key) {
        return Err(CandleBackendError::InvalidRequest(format!(
            "image.save image payload `{}` not found in store",
            payload_key.as_str()
        )));
    }

    let candle_image = backend.store().get_image(payload_key)?;
    let filename = build_safe_filename(
        prefix,
        run_id.as_str(),
        node_id.as_str(),
        backend.next_image_seq(),
    );
    let output_path = backend.output_dir().join(&filename);

    std::fs::create_dir_all(backend.output_dir()).map_err(|e| {
        CandleBackendError::InvalidRequest(format!(
            "image.save failed to create output directory: {e}"
        ))
    })?;

    validate_path_inside_output_dir(&output_path, backend.output_dir())?;

    let png_bytes = encode_tensor_to_png(&candle_image)?;
    std::fs::write(&output_path, &png_bytes).map_err(|e| {
        CandleBackendError::InvalidRequest(format!("image.save failed to write PNG file: {e}"))
    })?;

    Ok(artifact_ref_for_output_path(
        &output_path,
        backend.output_dir(),
    ))
}

fn artifact_ref_for_output_path(
    output_path: &std::path::Path,
    output_dir: &std::path::Path,
) -> ArtifactRef {
    let reference = output_path
        .strip_prefix(output_dir)
        .ok()
        .map(|relative| format!("output/{}", relative.to_string_lossy()))
        .unwrap_or_else(|| output_path.to_string_lossy().to_string());
    ArtifactRef::new(reference)
}

fn build_safe_filename(prefix: &str, run_id: &str, node_id: &str, seq: u64) -> String {
    let safe_prefix = sanitize_component(prefix, 64);
    let safe_run_id = sanitize_component(run_id, 128);
    let safe_node_id = sanitize_component(node_id, 128);
    format!(
        "{}_{}_{}_{}.png",
        safe_prefix, safe_run_id, safe_node_id, seq
    )
}

fn sanitize_component(input: &str, max_len: usize) -> String {
    let result: String = input
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();

    if result.is_empty() {
        return "_unnamed".to_string();
    }

    if result.chars().count() > max_len {
        result.chars().take(max_len).collect()
    } else {
        result
    }
}

fn validate_path_inside_output_dir(
    path: &std::path::Path,
    output_dir: &std::path::Path,
) -> Result<(), CandleBackendError> {
    let output_dir_canon = std::fs::canonicalize(output_dir).map_err(|_| {
        CandleBackendError::InvalidRequest(format!(
            "image.save output directory path does not exist or is inaccessible: {}",
            output_dir.display()
        ))
    })?;

    let parent = path.parent().unwrap_or(path);
    let parent_canon = std::fs::canonicalize(parent).map_err(|_| {
        CandleBackendError::InvalidRequest(format!(
            "image.save target path parent directory is inaccessible: {}",
            parent.display()
        ))
    })?;

    if !parent_canon.starts_with(&output_dir_canon) {
        return Err(CandleBackendError::InvalidRequest(format!(
            "image.save target path escapes workspace output directory: {}",
            path.display()
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_allows_alphanumeric_dash_underscore() {
        assert_eq!(sanitize_component("abc-123_xyz", 64), "abc-123_xyz");
    }

    #[test]
    fn sanitize_replaces_invalid_chars() {
        assert_eq!(sanitize_component("a/b:c*d", 64), "a_b_c_d");
    }

    #[test]
    fn sanitize_empty_becomes_unnamed() {
        assert_eq!(sanitize_component("", 64), "_unnamed");
    }

    #[test]
    fn sanitize_truncates_long_input() {
        let long = "a".repeat(200);
        let result = sanitize_component(&long, 64);
        assert_eq!(result.chars().count(), 64);
        assert!(result.chars().all(|c| c == 'a'));
    }

    #[test]
    fn build_safe_filename_format() {
        let filename = build_safe_filename("reimagine", "run-test", "node-a", 5);
        assert_eq!(filename, "reimagine_run-test_node-a_5.png");
    }

    #[test]
    fn build_safe_filename_sanitizes_path_traversal() {
        let filename = build_safe_filename("../../etc", "run", "node", 0);
        assert!(!filename.contains(".."));
        assert!(filename.contains("_"));
    }
}
