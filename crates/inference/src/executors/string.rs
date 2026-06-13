//! `builtin.string` executor.
//!
//! This node is a pure passthrough: it reads the `value` param and
//! emits it as the `value` output. It does not call the inference
//! backend.

use std::sync::Arc;

use reimagine_core::model::{ParamValue, SlotId};
use reimagine_runtime::{NodeExecutionContext, NodeExecutor, NodeExecutorError, RuntimeValue};

/// `builtin.string` executor. Pure param passthrough, no backend call.
pub struct StringExecutor;

#[async_trait::async_trait]
impl NodeExecutor for StringExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<(SlotId, Arc<RuntimeValue>)>, NodeExecutorError> {
        let value = context
            .params()
            .get(&SlotId::new("value"))
            .cloned()
            .unwrap_or(ParamValue::String(String::new()));
        Ok(vec![(
            SlotId::new("value"),
            Arc::new(RuntimeValue::Param(value)),
        )])
    }
}
