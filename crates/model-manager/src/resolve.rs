//! Model descriptor and readiness resolution facade.

mod descriptor;
mod readiness;

pub use descriptor::{
    ManifestModelResolver, ModelDescriptorResolver, ModelResolution, ResolvedComponent,
    ResolvedDescriptorView, ResolvedModelInfo,
};
pub use readiness::ModelReadinessResolver;
