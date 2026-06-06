//! Tests for TensorData — cheap-clone semantics and public API.

use reimagine_core::model::TensorData;

#[test]
fn from_vec_and_accessors() {
    let td = TensorData::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    assert_eq!(td.shape(), &[2, 3]);
    assert_eq!(td.numel(), 6);
    assert_eq!(td.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn to_vec_returns_copy() {
    let td = TensorData::from_vec(vec![1.0, 2.0], vec![2]);
    let v = td.to_vec();
    assert_eq!(v, vec![1.0, 2.0]);

    // to_vec is a copy — modifying it doesn't affect the original
    let mut v2 = td.to_vec();
    v2[0] = 99.0;
    assert_eq!(td.as_slice()[0], 1.0);
}

/// Cheap-clone semantics: cloning a TensorData must NOT deep-copy the data.
/// Public slices from a clone should point at the same immutable backing data.
#[test]
fn cheap_clone_semantics() {
    let big: Vec<f32> = (0..100_000).map(|i| i as f32).collect();
    let td = TensorData::from_vec(big, vec![100_000]);
    let cloned = td.clone();

    assert!(std::ptr::eq(td.as_slice().as_ptr(), cloned.as_slice().as_ptr()));
    assert_eq!(td, cloned);
    assert_eq!(cloned.as_slice().len(), 100_000);
}

/// PartialEq still works after the Arc migration.
#[test]
fn equality() {
    let a = TensorData::from_vec(vec![1.0, 2.0], vec![2]);
    let b = TensorData::from_vec(vec![1.0, 2.0], vec![2]);
    let c = TensorData::from_vec(vec![1.0, 3.0], vec![2]);

    assert_eq!(a, b);
    assert_ne!(a, c);
}

/// Debug output is available (doesn't need to be a specific format).
#[test]
fn debug_smoke() {
    let td = TensorData::from_vec(vec![1.0], vec![1]);
    let dbg = format!("{td:?}");
    assert!(!dbg.is_empty());
}
