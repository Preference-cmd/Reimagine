#![allow(unused_imports)]
//! Scheduler implementations for SDXL diffusion sampling.
//!
//! Currently contains the EulerNormalScheduler. Future schedulers (DDIM,
//! PNDM, etc.) can be added as sibling files and re-exported here.

mod euler_normal;

pub use euler_normal::EulerNormalScheduler;
