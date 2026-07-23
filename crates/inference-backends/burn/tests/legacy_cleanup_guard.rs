use std::fs;
use std::path::{Path, PathBuf};

const LEGACY_PATTERNS: &[&str] = &[
    "burn_ndarray",
    "NdArray",
    "BurnTensor::Ndarray",
    "ClipTextEncoderWeights",
    "ClipTransformerWeights",
    "ClipWeightData",
    "load_clip_l",
    "load_clip_g",
    "load_from_path(",
];

#[test]
fn production_sources_do_not_reference_legacy_burn_paths() {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src_root = crate_root.join("src");
    let mut violations = Vec::new();
    collect_legacy_references(&src_root, &mut violations);

    assert!(
        violations.is_empty(),
        "legacy Burn references remain in production sources:\n{}",
        violations.join("\n")
    );
}

fn collect_legacy_references(path: &Path, violations: &mut Vec<String>) {
    let entries = fs::read_dir(path).unwrap_or_else(|err| {
        panic!("failed to read {}: {err}", path.display());
    });

    for entry in entries {
        let entry = entry.expect("directory entry");
        let path = entry.path();
        if path.is_dir() {
            collect_legacy_references(&path, violations);
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }

        let source = fs::read_to_string(&path).unwrap_or_else(|err| {
            panic!("failed to read {}: {err}", path.display());
        });
        for (index, line) in source.lines().enumerate() {
            if LEGACY_PATTERNS.iter().any(|pattern| line.contains(pattern)) {
                violations.push(format!(
                    "{}:{}:{}",
                    path.strip_prefix(env!("CARGO_MANIFEST_DIR"))
                        .unwrap_or(&path)
                        .display(),
                    index + 1,
                    line.trim()
                ));
            }
        }
    }
}
