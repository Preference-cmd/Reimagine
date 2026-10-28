//! Kernel operations for collections:
//! - `collection.create`
//! - `collection.get`
//! - `collection.set_slot`
//! - `collection.undo`
//! - `collection.redo`

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use mod_sdk::{OperationAnnotations, OperationSpec};
use reimagine_core::board::{BoardItemKind, BoardItemPosition, BoardItemSize};
use reimagine_core::collection::Slot;
use reimagine_core::command::CommandStatus;
use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::model::{AssetId, BoardItemId, CollectionId, DiagnosticId, SlotId};
use serde_json::{Value, json};

use crate::board_service::BoardService;
use crate::collection_service::CollectionService;
use crate::error::AppHostError;
use crate::operation_service::{KernelCall, KernelOp, OperationService};

fn collection_caller_error(code: &str, message: impl Into<String>) -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new(format!("op:collection:{code}")),
        DiagnosticCode::new(format!("COLLECTION/CALLER_{code}")),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("collection_ops"),
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new("op.collection")),
    )
}

fn collection_internal_error(code: &str, message: impl Into<String>) -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new(format!("op:collection:{code}")),
        DiagnosticCode::new(format!("COLLECTION/INTERNAL_{code}")),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("collection_ops"),
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new("op.collection")),
    )
}

/// The built-in `collection.create` kernel operation.
pub struct CollectionCreateOp {
    spec: OperationSpec,
    collection_service: Arc<CollectionService>,
    board_service: Arc<BoardService>,
}

impl CollectionCreateOp {
    pub fn new(
        collection_service: Arc<CollectionService>,
        board_service: Arc<BoardService>,
    ) -> Self {
        Self {
            spec: OperationSpec {
                name: "collection.create".to_owned(),
                description: "Create a collection document and add an item to the board".to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "assets": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "format": "asset"
                            }
                        }
                    },
                    "required": ["name", "assets"],
                    "additionalProperties": false,
                    "examples": [{ "name": "Batch 1", "assets": ["asset-1", "asset-2"] }]
                }),
                output_schema: json!({
                    "type": "object",
                    "properties": {
                        "collection_id": { "type": "string" },
                        "board_item_id": { "type": "string" }
                    },
                    "required": ["collection_id", "board_item_id"],
                    "additionalProperties": false,
                    "examples": [{ "collection_id": "col-1", "board_item_id": "item-1" }]
                }),
                annotations: OperationAnnotations {
                    read_only: false,
                    destructive: false,
                    idempotent: false,
                    deterministic: true,
                    open_world: false,
                },
            },
            collection_service,
            board_service,
        }
    }
}

#[async_trait]
impl KernelOp for CollectionCreateOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    fn changes_documents(&self) -> bool {
        true
    }

    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
        let name = call
            .caller_input
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| collection_caller_error("INVALID_INPUT", "missing 'name' string"))?;

        let assets_arr = call
            .caller_input
            .get("assets")
            .and_then(Value::as_array)
            .ok_or_else(|| collection_caller_error("INVALID_INPUT", "missing 'assets' array"))?;

        let mut slots = Vec::new();
        for (i, asset_val) in assets_arr.iter().enumerate() {
            let asset_str = asset_val.as_str().ok_or_else(|| {
                collection_caller_error("INVALID_INPUT", "asset item must be string")
            })?;
            let asset_id = AssetId::try_new(asset_str)
                .map_err(|e| collection_caller_error("INVALID_ASSET_ID", e))?;
            let slot_id = SlotId::new(format!("slot-{}", i + 1));
            slots.push(Slot::new(slot_id, asset_id));
        }

        static ID_COUNTER: AtomicU64 = AtomicU64::new(1);
        let count = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default();
        let collection_id = CollectionId::new(format!("col-{nanos:x}-{count:x}"));
        let board_item_id = BoardItemId::new(format!("item-col-{nanos:x}-{count:x}"));

        // 1. Create and persist collection (no event emitted yet)
        let doc = self
            .collection_service
            .create(&call.project_id, &collection_id, name, slots)
            .await
            .map_err(|e| collection_internal_error("CREATE_FAILED", e.to_string()))?;

        // 2. Add collection item to project Board under lock
        let board_res = self
            .board_service
            .append_item(
                &call.project_id,
                &call.context,
                board_item_id.clone(),
                BoardItemKind::Collection {
                    collection_id: collection_id.clone(),
                },
                BoardItemPosition::new(0, 0),
                BoardItemSize::new(300, 200),
            )
            .await;

        match board_res {
            Ok(result) if result.status() == CommandStatus::Applied => {
                // 3. On success publish collection.changed
                self.collection_service.emit_changed(&doc);
            }
            Ok(_rejected) => {
                let _ = self
                    .collection_service
                    .remove(&call.project_id, &collection_id)
                    .await;
                return Err(collection_internal_error(
                    "BOARD_REJECTED",
                    "adding collection item to board was rejected",
                ));
            }
            Err(e) => {
                let _ = self
                    .collection_service
                    .remove(&call.project_id, &collection_id)
                    .await;
                return Err(collection_internal_error(
                    "BOARD_APPLY_FAILED",
                    e.to_string(),
                ));
            }
        }

        Ok(json!({
            "collection_id": collection_id.as_str(),
            "board_item_id": board_item_id.as_str(),
        }))
    }
}

/// The built-in `collection.get` kernel operation.
pub struct CollectionGetOp {
    spec: OperationSpec,
    collection_service: Arc<CollectionService>,
}

impl CollectionGetOp {
    pub fn new(collection_service: Arc<CollectionService>) -> Self {
        Self {
            spec: OperationSpec {
                name: "collection.get".to_owned(),
                description: "Get collection details and slots".to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "collection_id": { "type": "string" }
                    },
                    "required": ["collection_id"],
                    "additionalProperties": false,
                    "examples": [{ "collection_id": "col-1" }]
                }),
                output_schema: json!({
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "name": { "type": "string" },
                        "version": { "type": "integer", "minimum": 0 },
                        "slots": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": { "type": "string" },
                                    "asset_id": { "type": "string" }
                                },
                                "required": ["id", "asset_id"],
                                "additionalProperties": false
                            }
                        }
                    },
                    "required": ["id", "name", "version", "slots"],
                    "additionalProperties": false,
                    "examples": [{
                        "id": "col-1",
                        "name": "Batch 1",
                        "version": 0,
                        "slots": [{ "id": "slot-1", "asset_id": "asset-1" }]
                    }]
                }),
                annotations: OperationAnnotations {
                    read_only: true,
                    destructive: false,
                    idempotent: true,
                    deterministic: true,
                    open_world: false,
                },
            },
            collection_service,
        }
    }
}

#[async_trait]
impl KernelOp for CollectionGetOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
        let col_str = call
            .caller_input
            .get("collection_id")
            .and_then(Value::as_str)
            .ok_or_else(|| collection_caller_error("INVALID_INPUT", "missing 'collection_id'"))?;

        let collection_id = CollectionId::try_new(col_str)
            .map_err(|e| collection_caller_error("INVALID_COLLECTION_ID", e))?;

        let doc = self
            .collection_service
            .snapshot(&call.project_id, &collection_id)
            .await
            .map_err(|err| match err {
                AppHostError::UnknownCollection { .. } => collection_caller_error(
                    "UNKNOWN_COLLECTION",
                    format!("collection `{collection_id}` not found"),
                ),
                other => collection_internal_error("SNAPSHOT_FAILED", other.to_string()),
            })?;

        let slots: Vec<Value> = doc
            .slots()
            .iter()
            .map(|s| {
                json!({
                    "id": s.id.as_str(),
                    "asset_id": s.asset.as_str(),
                })
            })
            .collect();

        Ok(json!({
            "id": doc.id().as_str(),
            "name": doc.name(),
            "version": doc.version().get(),
            "slots": slots,
        }))
    }
}

/// The built-in `collection.set_slot` kernel operation.
pub struct CollectionSetSlotOp {
    spec: OperationSpec,
    collection_service: Arc<CollectionService>,
}

impl CollectionSetSlotOp {
    pub fn new(collection_service: Arc<CollectionService>) -> Self {
        Self {
            spec: OperationSpec {
                name: "collection.set_slot".to_owned(),
                description: "Change the asset pointed to by a slot in a collection".to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "collection_id": { "type": "string" },
                        "slot_id": { "type": "string" },
                        "asset": {
                            "type": "string",
                            "format": "asset"
                        }
                    },
                    "required": ["collection_id", "slot_id", "asset"],
                    "additionalProperties": false,
                    "examples": [{
                        "collection_id": "col-1",
                        "slot_id": "slot-1",
                        "asset": "asset-2"
                    }]
                }),
                output_schema: json!({
                    "type": "object",
                    "properties": {
                        "version": { "type": "integer", "minimum": 0 }
                    },
                    "required": ["version"],
                    "additionalProperties": false,
                    "examples": [{ "version": 1 }]
                }),
                annotations: OperationAnnotations {
                    read_only: false,
                    destructive: false,
                    idempotent: false,
                    deterministic: true,
                    open_world: false,
                },
            },
            collection_service,
        }
    }
}

#[async_trait]
impl KernelOp for CollectionSetSlotOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    fn changes_documents(&self) -> bool {
        true
    }

    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
        let col_str = call
            .caller_input
            .get("collection_id")
            .and_then(Value::as_str)
            .ok_or_else(|| collection_caller_error("INVALID_INPUT", "missing 'collection_id'"))?;
        let slot_str = call
            .caller_input
            .get("slot_id")
            .and_then(Value::as_str)
            .ok_or_else(|| collection_caller_error("INVALID_INPUT", "missing 'slot_id'"))?;
        let asset_str = call
            .caller_input
            .get("asset")
            .and_then(Value::as_str)
            .ok_or_else(|| collection_caller_error("INVALID_INPUT", "missing 'asset'"))?;

        let collection_id = CollectionId::try_new(col_str)
            .map_err(|e| collection_caller_error("INVALID_COLLECTION_ID", e))?;
        let slot_id =
            SlotId::try_new(slot_str).map_err(|e| collection_caller_error("INVALID_SLOT_ID", e))?;
        let asset_id = AssetId::try_new(asset_str)
            .map_err(|e| collection_caller_error("INVALID_ASSET_ID", e))?;

        // Verify collection existence and get current version
        let snapshot = self
            .collection_service
            .snapshot(&call.project_id, &collection_id)
            .await
            .map_err(|err| match err {
                AppHostError::UnknownCollection { .. } => collection_caller_error(
                    "UNKNOWN_COLLECTION",
                    format!("collection `{collection_id}` not found"),
                ),
                other => collection_internal_error("SNAPSHOT_FAILED", other.to_string()),
            })?;

        if snapshot.slot(&slot_id).is_none() {
            return Err(collection_caller_error(
                "UNKNOWN_SLOT",
                format!("slot `{slot_id}` not found in collection `{collection_id}`"),
            ));
        }

        let result = self
            .collection_service
            .set_slot_asset(
                &call.project_id,
                &collection_id,
                &call.context,
                slot_id,
                asset_id,
            )
            .await
            .map_err(|e| collection_internal_error("APPLY_FAILED", e.to_string()))?;

        if result.status() == CommandStatus::Rejected {
            if let Some(first) = result.diagnostics().first() {
                return Err(first.clone());
            }
            return Err(collection_caller_error("REJECTED", "command rejected"));
        }

        Ok(json!({ "version": result.version().get() }))
    }
}

/// The built-in `collection.undo` kernel operation.
pub struct CollectionUndoOp {
    spec: OperationSpec,
    collection_service: Arc<CollectionService>,
}

impl CollectionUndoOp {
    pub fn new(collection_service: Arc<CollectionService>) -> Self {
        Self {
            spec: OperationSpec {
                name: "collection.undo".to_owned(),
                description: "Undo the last collection modification".to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "collection_id": { "type": "string" }
                    },
                    "required": ["collection_id"],
                    "additionalProperties": false,
                    "examples": [{ "collection_id": "col-1" }]
                }),
                output_schema: json!({
                    "type": "object",
                    "properties": {
                        "applied": { "type": "boolean" },
                        "version": { "type": "integer", "minimum": 0 }
                    },
                    "required": ["applied", "version"],
                    "additionalProperties": false,
                    "examples": [{ "applied": true, "version": 1 }]
                }),
                annotations: OperationAnnotations {
                    read_only: false,
                    destructive: false,
                    idempotent: false,
                    deterministic: true,
                    open_world: false,
                },
            },
            collection_service,
        }
    }
}

#[async_trait]
impl KernelOp for CollectionUndoOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    fn changes_documents(&self) -> bool {
        true
    }

    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
        let col_str = call
            .caller_input
            .get("collection_id")
            .and_then(Value::as_str)
            .ok_or_else(|| collection_caller_error("INVALID_INPUT", "missing 'collection_id'"))?;

        let collection_id = CollectionId::try_new(col_str)
            .map_err(|e| collection_caller_error("INVALID_COLLECTION_ID", e))?;

        // Ensure collection exists
        let snapshot = self
            .collection_service
            .snapshot(&call.project_id, &collection_id)
            .await
            .map_err(|err| match err {
                AppHostError::UnknownCollection { .. } => collection_caller_error(
                    "UNKNOWN_COLLECTION",
                    format!("collection `{collection_id}` not found"),
                ),
                other => collection_internal_error("SNAPSHOT_FAILED", other.to_string()),
            })?;

        let maybe_res = self
            .collection_service
            .undo(&call.project_id, &collection_id)
            .await
            .map_err(|e| collection_internal_error("UNDO_FAILED", e.to_string()))?;

        let (applied, version) = match maybe_res {
            Some(res) => (true, res.version().get()),
            None => (false, snapshot.version().get()),
        };

        Ok(json!({ "applied": applied, "version": version }))
    }
}

/// The built-in `collection.redo` kernel operation.
pub struct CollectionRedoOp {
    spec: OperationSpec,
    collection_service: Arc<CollectionService>,
}

impl CollectionRedoOp {
    pub fn new(collection_service: Arc<CollectionService>) -> Self {
        Self {
            spec: OperationSpec {
                name: "collection.redo".to_owned(),
                description: "Redo the last undone collection modification".to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "collection_id": { "type": "string" }
                    },
                    "required": ["collection_id"],
                    "additionalProperties": false,
                    "examples": [{ "collection_id": "col-1" }]
                }),
                output_schema: json!({
                    "type": "object",
                    "properties": {
                        "applied": { "type": "boolean" },
                        "version": { "type": "integer", "minimum": 0 }
                    },
                    "required": ["applied", "version"],
                    "additionalProperties": false,
                    "examples": [{ "applied": true, "version": 1 }]
                }),
                annotations: OperationAnnotations {
                    read_only: false,
                    destructive: false,
                    idempotent: false,
                    deterministic: true,
                    open_world: false,
                },
            },
            collection_service,
        }
    }
}

#[async_trait]
impl KernelOp for CollectionRedoOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    fn changes_documents(&self) -> bool {
        true
    }

    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
        let col_str = call
            .caller_input
            .get("collection_id")
            .and_then(Value::as_str)
            .ok_or_else(|| collection_caller_error("INVALID_INPUT", "missing 'collection_id'"))?;

        let collection_id = CollectionId::try_new(col_str)
            .map_err(|e| collection_caller_error("INVALID_COLLECTION_ID", e))?;

        // Ensure collection exists
        let snapshot = self
            .collection_service
            .snapshot(&call.project_id, &collection_id)
            .await
            .map_err(|err| match err {
                AppHostError::UnknownCollection { .. } => collection_caller_error(
                    "UNKNOWN_COLLECTION",
                    format!("collection `{collection_id}` not found"),
                ),
                other => collection_internal_error("SNAPSHOT_FAILED", other.to_string()),
            })?;

        let maybe_res = self
            .collection_service
            .redo(&call.project_id, &collection_id)
            .await
            .map_err(|e| collection_internal_error("REDO_FAILED", e.to_string()))?;

        let (applied, version) = match maybe_res {
            Some(res) => (true, res.version().get()),
            None => (false, snapshot.version().get()),
        };

        Ok(json!({ "applied": applied, "version": version }))
    }
}

/// Register all collection operations with an [`OperationService`].
pub fn register_collection_ops(
    op_service: &OperationService,
    collection_service: Arc<CollectionService>,
    board_service: Arc<BoardService>,
) {
    op_service.register_kernel_op(Arc::new(CollectionCreateOp::new(
        Arc::clone(&collection_service),
        Arc::clone(&board_service),
    )));
    op_service.register_kernel_op(Arc::new(CollectionGetOp::new(Arc::clone(
        &collection_service,
    ))));
    op_service.register_kernel_op(Arc::new(CollectionSetSlotOp::new(Arc::clone(
        &collection_service,
    ))));
    op_service.register_kernel_op(Arc::new(CollectionUndoOp::new(Arc::clone(
        &collection_service,
    ))));
    op_service.register_kernel_op(Arc::new(CollectionRedoOp::new(collection_service)));
}
