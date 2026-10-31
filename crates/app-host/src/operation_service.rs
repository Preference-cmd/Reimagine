#![allow(clippy::result_large_err)]
//! Operation service managing kernel operations, domain mod operations,
//! execution pipelines, and previews.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use mod_sdk::{Invocation, ModRegistry, OperationAnnotations, OperationSpec, RESERVED_NAMESPACES};
use reimagine_config::AppPaths;
use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandProvenance, OperationContext,
};
use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::event::Timestamp;
use reimagine_core::health::Health;
use reimagine_core::model::{AssetId, DiagnosticId, HistoryEntryId, ProjectId, StepId};
use reimagine_core::step::{StepOutcome, StepRecord, StepRun};
use reimagine_core::task::CancelToken;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::RwLock;
use tokio::sync::Mutex;

use crate::activity::{ActivityKind, ActivityService};
use crate::asset_store::{AssetStore, matches_media_type_pattern};
use crate::error::AppHostError;
use crate::step_log::StepLog;

fn op_caller_error(op_name: &str, code: &str, message: impl Into<String>) -> Diagnostic {
    let namespace = op_name.split('.').next().unwrap_or("kernel");
    Diagnostic::new(
        DiagnosticId::new(format!("op:{op_name}:{code}")),
        DiagnosticCode::new(format!("{}/CALLER_{code}", namespace.to_uppercase())),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("operation_service"),
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new(format!("op.{op_name}"))),
    )
}

fn op_internal_error(op_name: &str, code: &str, message: impl Into<String>) -> Diagnostic {
    let namespace = op_name.split('.').next().unwrap_or("kernel");
    Diagnostic::new(
        DiagnosticId::new(format!("op:{op_name}:{code}")),
        DiagnosticCode::new(format!("{}/INTERNAL_{code}", namespace.to_uppercase())),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("operation_service"),
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new(format!("op.{op_name}"))),
    )
}

/// Output keys through which a board-mutating operation names the session
/// history entry it produced and the board item it touched. The activity
/// record reads them generically, so an operation opts in by adding the keys
/// to its output schema instead of by an operation-specific branch.
const OUTPUT_HISTORY_ENTRY_ID: &str = "history_entry_id";
const OUTPUT_BOARD_ITEM_ID: &str = "board_item_id";

/// Largest inline text payload, in UTF-8 bytes, that one operation call may
/// carry in its input or place in its output. The step record keeps the call's
/// input and output, so a larger payload must be passed as an asset instead.
pub const MAX_INLINE_TEXT_BYTES: usize = 262_144;

/// Sum the UTF-8 byte lengths of every populated inline-text field that
/// `schema` marks (`"format": "inline-text"`) in `value`.
fn inline_text_bytes(schema: &Value, value: &Value) -> usize {
    mod_sdk::schema::find_inline_text_fields(schema, value)
        .into_iter()
        .fold(0usize, |total, field| total.saturating_add(field.bytes))
}

fn output_history_entry_id(output: &Value) -> Option<HistoryEntryId> {
    output
        .get(OUTPUT_HISTORY_ENTRY_ID)
        .and_then(Value::as_str)
        .and_then(|id| HistoryEntryId::try_new(id).ok())
}

fn output_targets(output: &Value) -> Vec<String> {
    output
        .get(OUTPUT_BOARD_ITEM_ID)
        .and_then(Value::as_str)
        .map(|item_id| vec![item_id.to_owned()])
        .unwrap_or_default()
}

/// Validate whether a preview key matches ^[A-Za-z0-9_-]{1,64}$.
pub fn is_valid_preview_key(key: &str) -> bool {
    !key.is_empty()
        && key.len() <= 64
        && key
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// Validate a preview key, returning a caller diagnostic if invalid.
pub fn validate_preview_key(key: &str) -> Result<(), Diagnostic> {
    if is_valid_preview_key(key) {
        Ok(())
    } else {
        Err(op_caller_error(
            "kernel",
            "INVALID_PREVIEW_KEY",
            format!("preview key `{key}` must match ^[A-Za-z0-9_-]{{1,64}}$"),
        ))
    }
}

/// Call arguments passed to a [`KernelOp`].
#[derive(Debug, Clone)]
pub struct KernelCall {
    pub invocation: Invocation,
    pub project_id: ProjectId,
    pub context: OperationContext,
    pub caller_input: Value,
}

/// Trait implemented by built-in kernel operations.
#[async_trait]
pub trait KernelOp: Send + Sync {
    fn spec(&self) -> &OperationSpec;
    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic>;
    fn changes_documents(&self) -> bool {
        false
    }
}

/// The built-in `asset.import` kernel operation.
pub struct AssetImportOp {
    spec: OperationSpec,
}

impl Default for AssetImportOp {
    fn default() -> Self {
        Self::new()
    }
}

impl AssetImportOp {
    pub fn new() -> Self {
        Self {
            spec: OperationSpec {
                name: "asset.import".to_owned(),
                description: "Import external files into project assets".to_owned(),
                input_schema: serde_json::json!( {
                    "type": "object",
                    "properties": {
                        "paths": {
                            "type": "array",
                            "items": { "type": "string" }
                        }
                    },
                    "required": ["paths"],
                    "additionalProperties": false,
                    "examples": [{ "paths": ["/example/path.txt"] }]
                }),
                output_schema: serde_json::json!( {
                    "type": "object",
                    "properties": {
                        "assets": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "format": "asset"
                            }
                        }
                    },
                    "required": ["assets"],
                    "additionalProperties": false,
                    "examples": [{ "assets": ["asset-1"] }]
                }),
                annotations: OperationAnnotations {
                    read_only: false,
                    destructive: false,
                    idempotent: false,
                    deterministic: true,
                    open_world: false,
                },
            },
        }
    }
}

#[async_trait]
impl KernelOp for AssetImportOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
        let paths_val = call
            .invocation
            .input
            .get("paths")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                op_caller_error("asset.import", "INVALID_INPUT", "missing 'paths' array")
            })?;

        let mut asset_rel_paths = Vec::new();
        for (i, p) in paths_val.iter().enumerate() {
            let p_str = p.as_str().ok_or_else(|| {
                op_caller_error(
                    "asset.import",
                    "INVALID_INPUT",
                    "path item must be a string",
                )
            })?;
            let src = Path::new(p_str);
            if !src.exists() {
                return Err(op_caller_error(
                    "asset.import",
                    "FILE_NOT_FOUND",
                    format!("import source file does not exist: `{p_str}`"),
                ));
            }
            let file_name = src
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("import_file");
            let target_name = format!("{i}_{file_name}");
            let target_dest = call.invocation.output_dir.join(&target_name);
            tokio::fs::copy(src, &target_dest).await.map_err(|e| {
                op_internal_error(
                    "asset.import",
                    "IO",
                    format!(
                        "failed to copy `{p_str}` to `{}`: {e}",
                        target_dest.display()
                    ),
                )
            })?;
            asset_rel_paths.push(Value::String(target_name));
        }

        Ok(serde_json::json!({ "assets": asset_rel_paths }))
    }
}

/// The built-in `asset.list` kernel operation.
pub struct AssetListOp {
    spec: OperationSpec,
    paths: AppPaths,
}

impl AssetListOp {
    pub fn new(paths: AppPaths) -> Self {
        Self {
            spec: OperationSpec {
                name: "asset.list".to_owned(),
                description: "List assets imported into the project, newest first".to_owned(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
                output_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "assets": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": { "type": "string" },
                                    "media_type": { "type": "string" },
                                    "size": { "type": "integer", "minimum": 0 },
                                    "origin": { "type": "string" }
                                },
                                "required": ["id", "media_type", "size", "origin"],
                                "additionalProperties": false
                            }
                        }
                    },
                    "required": ["assets"],
                    "additionalProperties": false,
                    "examples": [{ "assets": [] }]
                }),
                annotations: OperationAnnotations {
                    read_only: true,
                    destructive: false,
                    idempotent: true,
                    deterministic: true,
                    open_world: false,
                },
            },
            paths,
        }
    }
}

#[async_trait]
impl KernelOp for AssetListOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
        let store = AssetStore::new(self.paths.project_dir(call.project_id.as_str()));
        let mut records = store.list().await?;
        records.reverse();
        let assets: Vec<Value> = records
            .iter()
            .map(|record| {
                serde_json::json!({
                    "id": record.id.as_str(),
                    "media_type": record.media_type.as_str(),
                    "size": record.size,
                    "origin": record.origin.as_str(),
                })
            })
            .collect();
        Ok(serde_json::json!({ "assets": assets }))
    }
}

/// Provider of an operation: kernel or a domain mod.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationProvider {
    Kernel,
    Mod { id: String },
}

/// Public descriptor of an available operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationInfo {
    pub spec: OperationSpec,
    pub provider: OperationProvider,
    pub health: Health,
}

/// Request parameters for invoking an operation.
#[derive(Debug, Clone)]
pub struct InvokeRequest {
    pub name: String,
    pub input: Value,
    pub context: OperationContext,
    pub cancel: CancelToken,
    /// Set when the call is one step of a dynamic workflow run.
    pub run: Option<StepRun>,
}

/// Result of an operation execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvokeResult {
    pub step: StepRecord,
    pub output: Value,
}

#[derive(Clone)]
struct ActivePreview {
    generation: u64,
    cancel: CancelToken,
}

/// Operation orchestration service.
pub struct OperationService {
    paths: AppPaths,
    mod_registry: Arc<ModRegistry>,
    kernel_ops: Arc<RwLock<HashMap<String, Arc<dyn KernelOp>>>>,
    preview_root: PathBuf,
    active_previews: Arc<Mutex<HashMap<String, ActivePreview>>>,
    next_preview_generation: Arc<AtomicU64>,
    asset_stores: Mutex<HashMap<ProjectId, Arc<AssetStore>>>,
    step_logs: Mutex<HashMap<ProjectId, Arc<StepLog>>>,
    preview_locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    activity: RwLock<Option<Arc<ActivityService>>>,
}

impl std::fmt::Debug for OperationService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OperationService")
            .field("paths", &self.paths)
            .field("preview_root", &self.preview_root)
            .finish_non_exhaustive()
    }
}

impl OperationService {
    /// Previews live under `<app tmp>/preview/<key>/`. Previews do not outlive
    /// the session, so any left from an earlier one are deleted here.
    pub fn new(paths: AppPaths, mod_registry: Arc<ModRegistry>) -> Self {
        let preview_root = paths.tmp_dir().join("preview");
        if preview_root.exists() {
            let _ = std::fs::remove_dir_all(&preview_root);
        }
        let _ = std::fs::create_dir_all(&preview_root);

        let mut ops: HashMap<String, Arc<dyn KernelOp>> = HashMap::new();
        let import_op = Arc::new(AssetImportOp::new());
        ops.insert(import_op.spec().name.clone(), import_op);
        let list_op = Arc::new(AssetListOp::new(paths.clone()));
        ops.insert(list_op.spec().name.clone(), list_op);

        Self {
            paths,
            mod_registry,
            kernel_ops: Arc::new(RwLock::new(ops)),
            preview_root,
            active_previews: Arc::new(Mutex::new(HashMap::new())),
            next_preview_generation: Arc::new(AtomicU64::new(1)),
            asset_stores: Mutex::new(HashMap::new()),
            step_logs: Mutex::new(HashMap::new()),
            preview_locks: Mutex::new(HashMap::new()),
            activity: RwLock::new(None),
        }
    }

    /// Attach the workspace activity log. Recorded operation steps append one
    /// `applied` record after they finish successfully.
    pub fn set_activity_service(&self, activity: Arc<ActivityService>) {
        *self.activity.write().expect("activity slot poisoned") = Some(activity);
    }

    async fn asset_store(&self, project: &ProjectId) -> Arc<AssetStore> {
        let mut stores = self.asset_stores.lock().await;
        stores
            .entry(project.clone())
            .or_insert_with(|| {
                let project_dir = self.paths.project_dir(project.as_str());
                Arc::new(AssetStore::new(project_dir))
            })
            .clone()
    }

    async fn step_log(&self, project: &ProjectId) -> Arc<StepLog> {
        let mut logs = self.step_logs.lock().await;
        logs.entry(project.clone())
            .or_insert_with(|| {
                let project_dir = self.paths.project_dir(project.as_str());
                Arc::new(StepLog::new(project_dir))
            })
            .clone()
    }

    /// Append one `applied` activity record for a successful recorded step.
    /// An activity failure is logged, not propagated: the step already ran.
    ///
    /// When the validated output carries a `history_entry_id` and/or a
    /// `board_item_id`, the record adopts them as its history entry and
    /// targets, so an undo derived from the record reverses the board change
    /// the operation made rather than the preceding entry.
    async fn record_op_activity(
        &self,
        project: &ProjectId,
        request: &InvokeRequest,
        output: &Value,
    ) {
        let activity = self
            .activity
            .read()
            .expect("activity slot poisoned")
            .clone();
        let Some(activity) = activity else {
            return;
        };

        if let Err(error) = activity
            .record(
                project,
                ActivityKind::Applied,
                request.context.actor().clone(),
                format!("Ran {}", request.name),
                "activity.op",
                serde_json::json!({ "op": request.name }),
                output_targets(output),
                output_history_entry_id(output),
            )
            .await
        {
            tracing::warn!(
                project_id = %project,
                op = %request.name,
                %error,
                "failed to append op activity record"
            );
        }
    }

    async fn preview_lock_for(&self, key: &str) -> Arc<Mutex<()>> {
        let mut locks = self.preview_locks.lock().await;
        locks
            .entry(key.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    /// Return the temporary directory for a preview key.
    pub fn preview_dir(&self, preview_key: &str) -> Result<PathBuf, Diagnostic> {
        validate_preview_key(preview_key)?;
        Ok(self.preview_root.join(preview_key))
    }

    /// Resolve and validate a relative file path inside a preview directory.
    pub fn resolve_preview_file(
        &self,
        preview_key: &str,
        relative_path: &str,
    ) -> Result<PathBuf, AppHostError> {
        let preview_dir = self.preview_dir(preview_key)?;
        Ok(validate_relative_path(&preview_dir, relative_path)?)
    }

    /// Register a kernel operation.
    pub fn register_kernel_op(&self, op: Arc<dyn KernelOp>) {
        let mut guard = self.kernel_ops.write().expect("kernel_ops poisoned");
        guard.insert(op.spec().name.clone(), op);
    }

    /// List all operations (kernel and mod) available for a project.
    pub async fn list(&self, _project: &ProjectId) -> Vec<OperationInfo> {
        let mut result = Vec::new();

        // 1. Kernel operations
        let kernel_guard = self.kernel_ops.read().expect("kernel_ops poisoned");
        for op in kernel_guard.values() {
            result.push(OperationInfo {
                spec: op.spec().clone(),
                provider: OperationProvider::Kernel,
                health: Health::Available,
            });
        }

        // 2. Mod operations
        for registered in self.mod_registry.list() {
            let is_reserved = RESERVED_NAMESPACES.contains(&registered.manifest.id.as_str());
            let health = if is_reserved {
                Health::Unavailable(op_caller_error(
                    &registered.manifest.id,
                    "RESERVED_NAMESPACE",
                    format!("mod `{}` uses reserved namespace", registered.manifest.id),
                ))
            } else {
                registered.health.clone()
            };

            for op in &registered.manifest.operations {
                result.push(OperationInfo {
                    spec: op.clone(),
                    provider: OperationProvider::Mod {
                        id: registered.manifest.id.clone(),
                    },
                    health: health.clone(),
                });
            }
        }

        result
    }

    /// Find an operation specification and its provider.
    async fn find_operation(
        &self,
        name: &str,
    ) -> Result<(OperationSpec, OperationProvider), Diagnostic> {
        let kernel_guard = self.kernel_ops.read().expect("kernel_ops poisoned");
        if let Some(op) = kernel_guard.get(name) {
            return Ok((op.spec().clone(), OperationProvider::Kernel));
        }
        drop(kernel_guard);

        for registered in self.mod_registry.list() {
            if RESERVED_NAMESPACES.contains(&registered.manifest.id.as_str()) {
                continue;
            }
            if let Some(spec) = registered
                .manifest
                .operations
                .iter()
                .find(|op| op.name == name)
            {
                return Ok((
                    spec.clone(),
                    OperationProvider::Mod {
                        id: registered.manifest.id.clone(),
                    },
                ));
            }
        }

        Err(op_caller_error(
            name,
            "UNKNOWN_OPERATION",
            format!("operation `{name}` is not registered"),
        ))
    }

    /// Invoke an operation following the 6-step flow.
    pub async fn invoke(
        &self,
        project: &ProjectId,
        request: InvokeRequest,
    ) -> Result<InvokeResult, Diagnostic> {
        let (spec, provider) = self.find_operation(&request.name).await?;

        // Step 1: Validate input against schema
        let validator = jsonschema::validator_for(&spec.input_schema).map_err(|e| {
            op_internal_error(
                &request.name,
                "INVALID_SCHEMA",
                format!("invalid input schema for `{}`: {e}", request.name),
            )
        })?;
        if let Some(violations) = mod_sdk::schema::violations(&validator, &request.input) {
            return Err(op_caller_error(
                &request.name,
                "INVALID_INPUT",
                format!(
                    "input validation failed for `{}`: {violations}",
                    request.name
                ),
            ));
        }

        // Inline text becomes part of the step record, so the caller must
        // switch to an asset above the cap. Reject before the scratch dir and
        // the start record exist, leaving no trace of an oversized call.
        let inline_input_bytes = inline_text_bytes(&spec.input_schema, &request.input);
        if inline_input_bytes > MAX_INLINE_TEXT_BYTES {
            return Err(op_caller_error(
                &request.name,
                "INLINE_TEXT_TOO_LARGE",
                format!(
                    "inline text input is {inline_input_bytes} bytes; the limit is \
                     {MAX_INLINE_TEXT_BYTES} bytes. Pass the text as an asset instead"
                ),
            ));
        }

        // Validate media types of input asset fields
        let project_dir = self.paths.project_dir(project.as_str());
        let asset_store = self.asset_store(project).await;
        let input_asset_fields =
            mod_sdk::schema::find_asset_fields(&spec.input_schema, &request.input);

        let mut asset_paths: Vec<(String, PathBuf)> = Vec::new();
        for field in &input_asset_fields {
            let val = request
                .input
                .pointer(&field.pointer)
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    op_caller_error(
                        &request.name,
                        "INVALID_INPUT",
                        format!("asset field at `{}` must be a string", field.pointer),
                    )
                })?;
            let asset_id = AssetId::try_new(val).map_err(|reason| {
                op_caller_error(
                    &request.name,
                    "INVALID_INPUT",
                    format!(
                        "asset field at `{}` holds an invalid asset id: {reason}",
                        field.pointer
                    ),
                )
            })?;
            let record = asset_store.get_asset(&asset_id).await?.ok_or_else(|| {
                op_caller_error(
                    &request.name,
                    "ASSET_NOT_FOUND",
                    format!("asset `{asset_id}` not found in project `{project}`"),
                )
            })?;

            if let Some(pattern) = &field.media_type
                && !matches_media_type_pattern(pattern, &record.media_type)
            {
                return Err(op_caller_error(
                    &request.name,
                    "MEDIA_TYPE_MISMATCH",
                    format!(
                        "asset `{asset_id}` media type `{}` does not match expected `{pattern}`",
                        record.media_type
                    ),
                ));
            }

            let abs_path = asset_store.content_path(&record.hash);
            if !abs_path.is_file() {
                return Err(op_internal_error(
                    &request.name,
                    "ASSET_FILE_MISSING",
                    format!(
                        "asset file missing on disk for asset `{asset_id}` at `{}`",
                        abs_path.display()
                    ),
                ));
            }
            asset_paths.push((field.pointer.clone(), abs_path));
        }

        // Step 2: Turn asset ids into read-only paths, create empty scratch dir, write "start" record
        let mut input_for_provider = request.input.clone();
        for (pointer, path) in asset_paths {
            set_json_pointer(
                &mut input_for_provider,
                &pointer,
                Value::String(path.to_string_lossy().to_string()),
            )?;
        }

        static STEP_COUNTER: AtomicU64 = AtomicU64::new(1);
        let count = STEP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default();
        let step_id = StepId::new(format!("step-{nanos:x}-{count:x}"));

        let scratch_dir = project_dir.join("scratch").join(step_id.as_str());
        if scratch_dir.exists() {
            let _ = tokio::fs::remove_dir_all(&scratch_dir).await;
        }
        tokio::fs::create_dir_all(&scratch_dir).await.map_err(|e| {
            op_internal_error(
                &request.name,
                "IO",
                format!(
                    "failed to create scratch dir `{}`: {e}",
                    scratch_dir.display()
                ),
            )
        })?;

        let op_version = match &provider {
            OperationProvider::Kernel => "0.1.0".to_owned(),
            OperationProvider::Mod { id } => self
                .mod_registry
                .list()
                .into_iter()
                .find(|m| m.manifest.id == *id)
                .map(|m| m.manifest.version.clone())
                .unwrap_or_else(|| "0.1.0".to_owned()),
        };

        let start_time = Timestamp::new(chrono::Utc::now().to_rfc3339());
        let mut step_record = StepRecord::new_running(
            step_id.clone(),
            request.name.clone(),
            op_version,
            request.input.clone(),
            request.context.clone(),
            start_time,
        );
        if let Some(run) = request.run.clone() {
            step_record = step_record.with_run(run);
        }

        // A dynamic workflow run needs a record for every step it executes,
        // even read-only ones: resume decides from these records.
        let should_record_step = request.run.is_some()
            || !spec.annotations.read_only
            || mod_sdk::schema::has_asset_field(&spec.output_schema);

        let step_log = self.step_log(project).await;
        if should_record_step {
            match step_log.record_start(&step_record).await {
                Ok(()) => {}
                Err(err) => {
                    let _ = tokio::fs::remove_dir_all(&scratch_dir).await;
                    return Err(err);
                }
            }
        }

        // Steps 3–5: Execute provider, validate output, and ingest output assets.
        // Once the start record is written, ANY failure must write a finish record
        // and clean up the scratch dir.
        let execution_result = async {
            // Step 3: Call the provider
            let invocation = Invocation::new(
                &request.name,
                input_for_provider,
                &scratch_dir,
                request.cancel.clone(),
            );

            let provider_result = match provider {
                OperationProvider::Kernel => {
                    let kernel_op = {
                        let kernel_guard = self.kernel_ops.read().expect("kernel_ops poisoned");
                        kernel_guard.get(&request.name).cloned().ok_or_else(|| {
                            op_caller_error(&request.name, "UNKNOWN_OPERATION", "kernel op missing")
                        })?
                    };

                    let call = KernelCall {
                        invocation,
                        project_id: project.clone(),
                        context: request.context.clone(),
                        caller_input: request.input.clone(),
                    };
                    kernel_op.invoke(call).await
                }
                OperationProvider::Mod { id } => self.mod_registry.invoke(&id, invocation).await,
            };

            let mut raw_output = provider_result?;

            // Step 4: Validate output schema
            let output_validator = jsonschema::validator_for(&spec.output_schema).map_err(|e| {
                op_internal_error(
                    &request.name,
                    "INVALID_SCHEMA",
                    format!("invalid output schema for `{}`: {e}", request.name),
                )
            })?;
            if let Some(violations) = mod_sdk::schema::violations(&output_validator, &raw_output) {
                return Err(op_internal_error(
                    &request.name,
                    "INVALID_OUTPUT",
                    format!(
                        "output validation failed for `{}`: {violations}",
                        request.name
                    ),
                ));
            }

            // The output lands in the step record too, so the same cap applies.
            // Reject before any asset is ingested or the step finishes.
            let inline_output_bytes = inline_text_bytes(&spec.output_schema, &raw_output);
            if inline_output_bytes > MAX_INLINE_TEXT_BYTES {
                return Err(op_internal_error(
                    &request.name,
                    "INVALID_OUTPUT",
                    format!(
                        "inline text output is {inline_output_bytes} bytes; the limit is \
                         {MAX_INLINE_TEXT_BYTES} bytes"
                    ),
                ));
            }

            // Step 5: Ingest the output's asset paths.
            // IMPORTANT: Validate ALL output asset paths before ingesting ANY of them,
            // so a bad subsequent path does not leave earlier assets registered against a failed step.
            let out_asset_fields =
                mod_sdk::schema::find_asset_fields(&spec.output_schema, &raw_output);
            let mut validated_assets = Vec::new();
            for field in &out_asset_fields {
                let rel_str = raw_output
                    .pointer(&field.pointer)
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        op_internal_error(
                            &request.name,
                            "INVALID_OUTPUT",
                            format!("output asset field at `{}` must be a string", field.pointer),
                        )
                    })?;

                let abs_file_path = validate_relative_path(&scratch_dir, rel_str)?;
                validated_assets.push((field, abs_file_path));
            }

            for (field, abs_file_path) in validated_assets {
                let record = asset_store
                    .ingest_file(step_id.clone(), &abs_file_path, field.media_type.as_deref())
                    .await?;

                set_json_pointer(
                    &mut raw_output,
                    &field.pointer,
                    Value::String(record.id.to_string()),
                )?;
            }

            Ok::<Value, Diagnostic>(raw_output)
        }
        .await;

        // Step 6: Finalize step record and delete scratch dir on every exit path
        let finish_time = Timestamp::new(chrono::Utc::now().to_rfc3339());
        let _ = tokio::fs::remove_dir_all(&scratch_dir).await;

        match execution_result {
            Ok(output) => {
                step_record.finish(
                    StepOutcome::Succeeded {
                        output: output.clone(),
                    },
                    finish_time,
                );
                if should_record_step {
                    step_log.record_finish(&step_record).await?;
                    self.record_op_activity(project, &request, &output).await;
                }

                Ok(InvokeResult {
                    step: step_record,
                    output,
                })
            }
            Err(err) => {
                let outcome = if request.cancel.is_cancelled() {
                    StepOutcome::Cancelled
                } else {
                    StepOutcome::failed(err.clone())
                };
                step_record.finish(outcome, finish_time);
                if should_record_step {
                    let _ = step_log.record_finish(&step_record).await;
                }

                Err(err)
            }
        }
    }

    /// Cancel an in-flight preview registered under `preview_key`, if any.
    ///
    /// The preview directory is left in place: the cancelled provider may still
    /// be writing to it, so the next preview under the same key or the next
    /// session's temporary-directory cleanup removes it. Cancelling a key that
    /// is not in flight is a no-op.
    pub async fn cancel_preview(&self, preview_key: &str) -> Result<(), Diagnostic> {
        validate_preview_key(preview_key)?;
        let mut active = self.active_previews.lock().await;
        if let Some(entry) = active.remove(preview_key) {
            entry.cancel.cancel();
        }
        Ok(())
    }

    /// Preview an operation into an ephemeral preview directory without recording steps or assets.
    pub async fn preview(
        &self,
        project: &ProjectId,
        name: &str,
        input: Value,
        preview_key: &str,
        cancel: CancelToken,
    ) -> Result<Value, Diagnostic> {
        validate_preview_key(preview_key)?;
        let (spec, provider) = self.find_operation(name).await?;

        // Refuse open_world ops
        if spec.annotations.open_world {
            return Err(op_caller_error(
                name,
                "OPEN_WORLD_PREVIEW_REFUSED",
                "preview is not supported for open-world operations",
            ));
        }

        // Refuse operations that change documents
        if let OperationProvider::Kernel = provider {
            let kernel_guard = self.kernel_ops.read().expect("kernel_ops poisoned");
            if let Some(op) = kernel_guard.get(name)
                && op.changes_documents()
            {
                return Err(op_caller_error(
                    name,
                    "DOCUMENT_CHANGE_PREVIEW_REFUSED",
                    "preview is not supported for operations that mutate documents",
                ));
            }
        }

        let generation = self.next_preview_generation.fetch_add(1, Ordering::Relaxed);
        let preview_dir = self.preview_dir(preview_key)?;

        // Cancel previous preview with the same key
        {
            let mut active = self.active_previews.lock().await;
            if let Some(previous) = active.remove(preview_key) {
                previous.cancel.cancel();
            }
            active.insert(
                preview_key.to_string(),
                ActivePreview {
                    generation,
                    cancel: cancel.clone(),
                },
            );
        }

        // The delete-and-recreate of <preview root>/<key>/ must happen only while this
        // preview is still the current generation, under a per-key async lock.
        let key_lock = self.preview_lock_for(preview_key).await;
        {
            let _guard = key_lock.lock().await;
            let is_current = {
                let active = self.active_previews.lock().await;
                active
                    .get(preview_key)
                    .is_some_and(|entry| entry.generation == generation)
            };
            if !is_current || cancel.is_cancelled() {
                return Err(op_caller_error(
                    name,
                    "CANCELLED",
                    "preview was cancelled or replaced",
                ));
            }

            if preview_dir.exists() {
                let _ = tokio::fs::remove_dir_all(&preview_dir).await;
            }
            tokio::fs::create_dir_all(&preview_dir).await.map_err(|e| {
                op_internal_error(
                    name,
                    "IO",
                    format!(
                        "failed to create preview dir `{}`: {e}",
                        preview_dir.display()
                    ),
                )
            })?;
        }

        let run_preview = async {
            // Step 1: Validate input
            let validator = jsonschema::validator_for(&spec.input_schema).map_err(|e| {
                op_internal_error(
                    name,
                    "INVALID_SCHEMA",
                    format!("invalid input schema for `{name}`: {e}"),
                )
            })?;
            if let Some(violations) = mod_sdk::schema::violations(&validator, &input) {
                return Err(op_caller_error(
                    name,
                    "INVALID_INPUT",
                    format!("input validation failed for `{name}`: {violations}"),
                ));
            }

            // A preview enters the same provider as an invocation, so it takes
            // the same input cap; the caller must move oversized text to an asset.
            let inline_input_bytes = inline_text_bytes(&spec.input_schema, &input);
            if inline_input_bytes > MAX_INLINE_TEXT_BYTES {
                return Err(op_caller_error(
                    name,
                    "INLINE_TEXT_TOO_LARGE",
                    format!(
                        "inline text input is {inline_input_bytes} bytes; the limit is \
                         {MAX_INLINE_TEXT_BYTES} bytes. Pass the text as an asset instead"
                    ),
                ));
            }

            // Resolve input asset IDs
            let asset_store = self.asset_store(project).await;
            let input_asset_fields = mod_sdk::schema::find_asset_fields(&spec.input_schema, &input);

            let mut asset_paths: Vec<(String, PathBuf)> = Vec::new();
            for field in &input_asset_fields {
                let val = input
                    .pointer(&field.pointer)
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        op_caller_error(
                            name,
                            "INVALID_INPUT",
                            format!("asset field at `{}` must be a string", field.pointer),
                        )
                    })?;
                let asset_id = AssetId::try_new(val).map_err(|reason| {
                    op_caller_error(
                        name,
                        "INVALID_INPUT",
                        format!(
                            "asset field at `{}` holds an invalid asset id: {reason}",
                            field.pointer
                        ),
                    )
                })?;
                let record = asset_store.get_asset(&asset_id).await?.ok_or_else(|| {
                    op_caller_error(
                        name,
                        "ASSET_NOT_FOUND",
                        format!("asset `{asset_id}` not found in project `{project}`"),
                    )
                })?;

                if let Some(pattern) = &field.media_type
                    && !matches_media_type_pattern(pattern, &record.media_type)
                {
                    return Err(op_caller_error(
                        name,
                        "MEDIA_TYPE_MISMATCH",
                        format!(
                            "asset `{asset_id}` media type `{}` does not match expected `{pattern}`",
                            record.media_type
                        ),
                    ));
                }

                let abs_path = asset_store.content_path(&record.hash);
                asset_paths.push((field.pointer.clone(), abs_path));
            }

            let mut input_for_provider = input.clone();
            for (pointer, path) in asset_paths {
                set_json_pointer(
                    &mut input_for_provider,
                    &pointer,
                    Value::String(path.to_string_lossy().to_string()),
                )?;
            }

            // Call provider
            let invocation = Invocation::new(name, input_for_provider, &preview_dir, cancel.clone());
            let provider_result = match provider {
                OperationProvider::Kernel => {
                    let kernel_op = {
                        let kernel_guard = self.kernel_ops.read().expect("kernel_ops poisoned");
                        kernel_guard.get(name).cloned().ok_or_else(|| {
                            op_caller_error(name, "UNKNOWN_OPERATION", "kernel op missing")
                        })?
                    };

                    let call = KernelCall {
                        invocation,
                        project_id: project.clone(),
                        context: OperationContext::new(
                            CommandActor::new(CommandActorKind::Human),
                            CommandProvenance::Direct,
                        ),
                        caller_input: input.clone(),
                    };
                    kernel_op.invoke(call).await
                }
                OperationProvider::Mod { id } => self.mod_registry.invoke(&id, invocation).await,
            };

            let mut raw_output = provider_result?;

            // Validate output schema
            let output_validator = jsonschema::validator_for(&spec.output_schema).map_err(|e| {
                op_internal_error(
                    name,
                    "INVALID_SCHEMA",
                    format!("invalid output schema for `{name}`: {e}"),
                )
            })?;
            if let Some(violations) = mod_sdk::schema::violations(&output_validator, &raw_output) {
                return Err(op_internal_error(
                    name,
                    "INVALID_OUTPUT",
                    format!("output validation failed for `{name}`: {violations}"),
                ));
            }

            // Replace relative asset paths with preview URIs
            let out_asset_fields =
                mod_sdk::schema::find_asset_fields(&spec.output_schema, &raw_output);
            for field in out_asset_fields {
                let rel_str = raw_output
                    .pointer(&field.pointer)
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        op_internal_error(
                            name,
                            "INVALID_OUTPUT",
                            format!("output asset field at `{}` must be a string", field.pointer),
                        )
                    })?;

                let _ = validate_relative_path(&preview_dir, rel_str)?;
                let uri = format!("reimagine-asset://preview/{preview_key}/{rel_str}");
                set_json_pointer(&mut raw_output, &field.pointer, Value::String(uri))?;
            }

            Ok::<Value, Diagnostic>(raw_output)
        }
        .await;

        match run_preview {
            Ok(output) => {
                // Succeeded previews leave their directory on disk for HTTP serving.
                // Remove from active_previews only if still current.
                let mut active = self.active_previews.lock().await;
                if let Some(entry) = active.get(preview_key)
                    && entry.generation == generation
                {
                    active.remove(preview_key);
                }
                Ok(output)
            }
            Err(err) => {
                // On exit due to error/cancel: remove entry and dir ONLY IF still current generation
                let is_current = {
                    let mut active = self.active_previews.lock().await;
                    if let Some(entry) = active.get(preview_key)
                        && entry.generation == generation
                    {
                        active.remove(preview_key);
                        true
                    } else {
                        false
                    }
                };
                if is_current {
                    let _guard = key_lock.lock().await;
                    let is_still_current = {
                        let active = self.active_previews.lock().await;
                        !active.contains_key(preview_key)
                    };
                    if is_still_current {
                        let _ = tokio::fs::remove_dir_all(&preview_dir).await;
                    }
                }
                Err(err)
            }
        }
    }
}

fn validate_relative_path(output_dir: &Path, rel: &str) -> Result<PathBuf, Diagnostic> {
    if rel.is_empty() {
        return Err(op_caller_error(
            "kernel",
            "INVALID_PATH",
            "asset path is empty",
        ));
    }
    let path = Path::new(rel);
    if path.is_absolute() {
        return Err(op_caller_error(
            "kernel",
            "ESCAPE_OUTPUT_DIR",
            "asset path must be relative",
        ));
    }
    for comp in path.components() {
        match comp {
            std::path::Component::ParentDir => {
                return Err(op_caller_error(
                    "kernel",
                    "ESCAPE_OUTPUT_DIR",
                    format!("path `{rel}` contains parent directory component `..`"),
                ));
            }
            std::path::Component::Prefix(_) | std::path::Component::RootDir => {
                return Err(op_caller_error(
                    "kernel",
                    "ESCAPE_OUTPUT_DIR",
                    "asset path must be relative",
                ));
            }
            std::path::Component::CurDir | std::path::Component::Normal(_) => {}
        }
    }

    let mut current = output_dir.to_path_buf();
    for comp in path.components() {
        if let std::path::Component::Normal(c) = comp {
            current.push(c);
            let meta = match std::fs::symlink_metadata(&current) {
                Ok(m) => m,
                Err(e) => {
                    return Err(op_caller_error(
                        "kernel",
                        "FILE_NOT_FOUND",
                        format!("asset file `{rel}` does not exist in output dir: {e}"),
                    ));
                }
            };
            if meta.file_type().is_symlink() {
                return Err(op_caller_error(
                    "kernel",
                    "SYMLINK_REJECTED",
                    format!(
                        "asset file `{rel}` component `{}` is a symlink",
                        c.to_string_lossy()
                    ),
                ));
            }
        }
    }

    let target = current;
    let meta = match std::fs::symlink_metadata(&target) {
        Ok(m) => m,
        Err(e) => {
            return Err(op_caller_error(
                "kernel",
                "FILE_NOT_FOUND",
                format!("asset file `{rel}` does not exist in output dir: {e}"),
            ));
        }
    };
    if meta.file_type().is_symlink() {
        return Err(op_caller_error(
            "kernel",
            "SYMLINK_REJECTED",
            format!("asset file `{rel}` is a symlink"),
        ));
    }
    if !meta.is_file() {
        return Err(op_caller_error(
            "kernel",
            "INVALID_FILE",
            format!("asset file `{rel}` is not a regular file"),
        ));
    }

    let canonical_target = target.canonicalize().map_err(|e| {
        op_internal_error(
            "kernel",
            "IO",
            format!("failed to canonicalize `{}`: {e}", target.display()),
        )
    })?;
    let canonical_out = output_dir.canonicalize().map_err(|e| {
        op_internal_error(
            "kernel",
            "IO",
            format!("failed to canonicalize `{}`: {e}", output_dir.display()),
        )
    })?;
    if !canonical_target.starts_with(&canonical_out) {
        return Err(op_caller_error(
            "kernel",
            "ESCAPE_OUTPUT_DIR",
            format!("asset path `{rel}` escapes output directory"),
        ));
    }

    Ok(target)
}

fn set_json_pointer(target: &mut Value, pointer: &str, new_val: Value) -> Result<(), Diagnostic> {
    if pointer.is_empty() {
        *target = new_val;
        return Ok(());
    }

    let tokens: Vec<String> = pointer
        .split('/')
        .skip(1)
        .map(|t| t.replace("~1", "/").replace("~0", "~"))
        .collect();

    if tokens.is_empty() {
        *target = new_val;
        return Ok(());
    }

    let mut current = target;
    for token in &tokens[..tokens.len() - 1] {
        match current {
            Value::Object(map) => {
                current = map.get_mut(token).ok_or_else(|| {
                    op_internal_error(
                        "kernel",
                        "POINTER_NOT_FOUND",
                        format!("token `{token}` not found in pointer `{pointer}`"),
                    )
                })?;
            }
            Value::Array(arr) => {
                let idx: usize = token.parse().map_err(|_| {
                    op_internal_error(
                        "kernel",
                        "POINTER_INVALID_INDEX",
                        format!("invalid array index `{token}` in pointer `{pointer}`"),
                    )
                })?;
                current = arr.get_mut(idx).ok_or_else(|| {
                    op_internal_error(
                        "kernel",
                        "POINTER_INDEX_OUT_OF_BOUNDS",
                        format!("index `{idx}` out of bounds in pointer `{pointer}`"),
                    )
                })?;
            }
            _ => {
                return Err(op_internal_error(
                    "kernel",
                    "POINTER_NOT_CONTAINER",
                    format!("cannot traverse through non-container at `{token}` in `{pointer}`"),
                ));
            }
        }
    }

    let last = &tokens[tokens.len() - 1];
    match current {
        Value::Object(map) => {
            map.insert(last.clone(), new_val);
        }
        Value::Array(arr) => {
            let idx: usize = last.parse().map_err(|_| {
                op_internal_error(
                    "kernel",
                    "POINTER_INVALID_INDEX",
                    format!("invalid array index `{last}` in pointer `{pointer}`"),
                )
            })?;
            if idx < arr.len() {
                arr[idx] = new_val;
            } else {
                return Err(op_internal_error(
                    "kernel",
                    "POINTER_INDEX_OUT_OF_BOUNDS",
                    format!("index `{idx}` out of bounds in pointer `{pointer}`"),
                ));
            }
        }
        _ => {
            return Err(op_internal_error(
                "kernel",
                "POINTER_NOT_CONTAINER",
                format!("cannot set property `{last}` on non-container in `{pointer}`"),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use mod_sdk::ModRegistry;
    use reimagine_config::AppPaths;

    struct SlowPreviewOp {
        spec: OperationSpec,
        started: Arc<tokio::sync::Notify>,
    }

    #[async_trait]
    impl KernelOp for SlowPreviewOp {
        fn spec(&self) -> &OperationSpec {
            &self.spec
        }

        async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
            // Leave a file behind so the test can prove cancellation does not
            // delete a directory that may still be written to.
            let marker = call.invocation.output_dir.join("in-flight.txt");
            tokio::fs::write(&marker, b"partial preview")
                .await
                .expect("write in-flight marker");
            self.started.notify_one();
            call.invocation.cancel.cancelled().await;
            Err(op_caller_error(
                "test.slow",
                "CANCELLED",
                "preview cancelled",
            ))
        }
    }

    fn slow_preview_service(
        dir: &tempfile::TempDir,
    ) -> (Arc<OperationService>, Arc<tokio::sync::Notify>) {
        let service = Arc::new(OperationService::new(
            AppPaths::new(dir.path()),
            Arc::new(ModRegistry::empty()),
        ));
        let started = Arc::new(tokio::sync::Notify::new());
        service.register_kernel_op(Arc::new(SlowPreviewOp {
            spec: OperationSpec {
                name: "test.slow".to_owned(),
                description: "slow preview".to_owned(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
                output_schema: serde_json::json!({ "type": "object", "properties": {} }),
                annotations: OperationAnnotations {
                    read_only: false,
                    destructive: false,
                    idempotent: false,
                    deterministic: true,
                    open_world: false,
                },
            },
            started: started.clone(),
        }));
        (service, started)
    }

    #[tokio::test]
    async fn cancel_preview_stops_the_provider_and_removes_the_entry() {
        let dir = tempfile::tempdir().expect("temp dir");
        let (service, started) = slow_preview_service(&dir);
        let preview_key = "cancel-key";

        let preview_service = Arc::clone(&service);
        let handle = tokio::spawn(async move {
            preview_service
                .preview(
                    &ProjectId::new("proj-cancel"),
                    "test.slow",
                    serde_json::json!({}),
                    preview_key,
                    CancelToken::new(),
                )
                .await
        });

        started.notified().await;
        service
            .cancel_preview(preview_key)
            .await
            .expect("cancel by key");

        assert!(
            service
                .active_previews
                .lock()
                .await
                .get(preview_key)
                .is_none(),
            "a cancelled preview must leave no active entry"
        );

        let error = handle
            .await
            .expect("join preview task")
            .expect_err("the cancelled preview must fail");
        assert_eq!(error.code().as_str(), "TEST/CALLER_CANCELLED");

        let preview_dir = service.preview_dir(preview_key).expect("valid key");
        assert!(
            preview_dir.join("in-flight.txt").is_file(),
            "cancel must not delete a directory an in-flight provider may still write"
        );
    }

    #[tokio::test]
    async fn cancel_preview_rejects_an_invalid_key_and_ignores_an_unknown_key() {
        let dir = tempfile::tempdir().expect("temp dir");
        let (service, _started) = slow_preview_service(&dir);

        let error = service
            .cancel_preview("../escape")
            .await
            .expect_err("invalid keys must be rejected");
        assert_eq!(error.code().as_str(), "KERNEL/CALLER_INVALID_PREVIEW_KEY");

        service
            .cancel_preview("not-in-flight")
            .await
            .expect("cancelling an idle key is a no-op");
    }
}
