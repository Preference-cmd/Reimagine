use reimagine_core::model::ModelRef;

use super::descriptor::{ModelResolution, ResolvedModelInfo};

#[allow(async_fn_in_trait)]
pub trait ModelReadinessResolver {
    async fn resolve_readiness(&self, model_ref: &ModelRef) -> ModelResolution<ResolvedModelInfo>;
}
