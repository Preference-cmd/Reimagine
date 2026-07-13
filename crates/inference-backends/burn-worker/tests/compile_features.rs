//! Compile-fail tests for the MB03 feature contract.
//!
//! The worker binary requires exactly one compute backend feature:
//! `wgpu` or `flex`. Zero or dual features produce intentional
//! build failures enforced by `build.rs`.
//!
//! These tests use `cargo check` as a subprocess to verify
//! the build script rejects illegal feature combinations.

use std::process::Command;

/// Helper: run `cargo check` for the burn-worker crate with the
/// given feature flags and return the exit status.
fn cargo_check(features: &[&str]) -> std::process::Output {
    let mut cmd = Command::new("cargo");
    cmd.arg("check");
    cmd.arg("-p");
    cmd.arg("reimagine-inference-burn-worker");
    cmd.arg("--no-default-features");
    if !features.is_empty() {
        cmd.arg(format!("--features={}", features.join(",")));
    }
    cmd.output().expect("failed to run cargo check")
}

#[test]
fn zero_features_fails_to_compile() {
    let output = cargo_check(&[]);
    assert!(!output.status.success(), "zero-feature build should fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("requires exactly one"),
        "stderr should mention exactly-one requirement: {stderr}"
    );
}

#[test]
fn dual_features_fails_to_compile() {
    let output = cargo_check(&["wgpu", "flex"]);
    assert!(!output.status.success(), "dual-feature build should fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("must not enable both"),
        "stderr should mention dual-feature rejection: {stderr}"
    );
}

#[test]
fn wgpu_only_builds_successfully() {
    let output = cargo_check(&["wgpu"]);
    assert!(
        output.status.success(),
        "wgpu-only build should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn flex_only_builds_successfully() {
    let output = cargo_check(&["flex"]);
    assert!(
        output.status.success(),
        "flex-only build should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
