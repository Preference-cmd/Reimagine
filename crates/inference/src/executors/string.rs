//! `builtin.string` executor.
//!
//! This node is a pure passthrough: it reads the `value` param and
//! emits it as the `value` output. It does not call the inference
//! backend.
//!
//! Retention: the passthrough value is declared `RunScoped` (the V1
//! default for executor outputs that aren't model handles).

use std::sync::Arc;

use reimagine_core::model::{ParamValue, SlotId};
use reimagine_inference_core::ExecutionOutput;

use crate::executor::{NodeExecutionContext, NodeExecutor, NodeExecutorError};

/// `builtin.string` executor. Pure param passthrough, no backend call.
pub struct StringExecutor;

#[async_trait::async_trait]
impl NodeExecutor for StringExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<ExecutionOutput>, NodeExecutorError> {
        let value = context
            .params()
            .get(&SlotId::new("value"))
            .cloned()
            .unwrap_or(ParamValue::String(String::new()));
        Ok(vec![ExecutionOutput::run_scoped(
            SlotId::new("value"),
            Arc::new(reimagine_inference_core::ExecutionValue::Param(value)),
        )])
    }
}
