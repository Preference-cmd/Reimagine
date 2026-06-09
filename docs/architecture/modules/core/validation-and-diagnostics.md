# Core Validation and Diagnostics

> Status: working draft
> Owner: `crates/core`

## Validation Layers

Core validation has three layers:

```text
Parse Validation
  JSON can be parsed into a workflow candidate

Structural Validation
  workflow is a legal editable canonical graph

Executable Readiness
  workflow can produce an execution plan for the selected run targets
```

Structural errors reject command batches. Readiness errors block execution only.

## Parse Validation

Parse validation happens when opening workflow JSON or importing a workflow:

- unsupported `schema_version`;
- invalid field types;
- invalid tagged `ParamValue`;
- malformed IDs;
- malformed endpoint shape.

If parse validation fails, core may return file-level diagnostics without constructing a `Workflow`.

## Structural Validation

Structural validation decides whether a workflow can enter or remain in a canonical editing session.

Structural checks:

- workflow id exists;
- node ids are unique;
- edge ids are unique;
- edge endpoint nodes exist;
- edge endpoint slots exist;
- `from` endpoints reference output slots or workflow inputs;
- `to` endpoints reference input slots or workflow outputs;
- slot kinds are compatible;
- `dynamic=true` input slots do not appear in `WorkflowNode.params`;
- params reference existing input slots;
- params values are compatible with slot kinds;
- each input slot has at most one incoming edge in V1;
- layout references existing nodes for command application.

Dangling layout found while loading an older file may be downgraded to a warning and normalization candidate, but commands should not create dangling layout.

## Executable Readiness

Readiness decides whether core can produce an execution plan. It does not block saving or editing.

Readiness is evaluated over the execution subgraph, not necessarily the whole graph.

Run target selection in V1:

```text
RunTargetSelection
  AllDefaultTargets
  ExplicitTargets(Vec<RunTarget>)
```

Default run targets are target-capable nodes: nodes whose `NodeDef` has no `required=true` output slots. This allows terminal nodes such as preview and save nodes to be run without requiring an output socket.

Explicit targets may select a target-capable node, a specific node output, or a workflow output. This supports partial execution of one selected target as well as full default-target execution.

The execution subgraph is found by tracing upstream dependencies from the selected run targets. Multiple selected targets produce one merged execution subgraph, so shared upstream nodes execute once.

Readiness checks:

- at least one run target exists;
- explicit targets exist and are valid;
- required input slots in the execution subgraph resolve to effective values;
- required outputs on pure nodes are consumed or exposed;
- pure nodes contribute to a run target;
- execution subgraph has no cycles;
- model refs may be checked by an external readiness capability;
- output paths and artifact destinations are valid when they are required for execution.

Cycle detection is executable readiness, not structural validity. Cyclic graphs may be saved and edited, but cannot run.

## Validation API Boundary

`core-workflow/01` should introduce the structural validation entry point without implementing the full command/session system.

Minimum API shape:

```text
StructuralValidator
  validate(workflow, node_catalog) -> OperationReport

ValidationReport = OperationReport
```

The validator should produce diagnostics using the existing core diagnostic model. A structurally valid workflow returns an empty or non-error report. A structurally invalid workflow can still be represented as parsed data, but `WorkflowSession` must later reject command batches that would commit structural errors.

`core-workflow/01` should cover structural checks that require only `Workflow`, `NodeDef`, and `NodeCatalog`. It should not call model-manager or runtime.

`core-workflow/03` should introduce executable readiness and execution-plan construction after command/session semantics are stable.

## Diagnostic Codes

Use stable, namespaced diagnostic codes. Initial workflow structural codes:

```text
CORE/WORKFLOW_SCHEMA_UNSUPPORTED
CORE/WORKFLOW_NODE_ID_DUPLICATE
CORE/WORKFLOW_EDGE_ID_DUPLICATE
CORE/WORKFLOW_NODE_TYPE_UNKNOWN
CORE/WORKFLOW_ENDPOINT_NODE_MISSING
CORE/WORKFLOW_ENDPOINT_SLOT_MISSING
CORE/WORKFLOW_ENDPOINT_DIRECTION_INVALID
CORE/WORKFLOW_SLOT_KIND_MISMATCH
CORE/WORKFLOW_PARAM_SLOT_MISSING
CORE/WORKFLOW_PARAM_ON_DYNAMIC_SLOT
CORE/WORKFLOW_PARAM_KIND_MISMATCH
CORE/WORKFLOW_INPUT_EDGE_DUPLICATE
CORE/WORKFLOW_LAYOUT_NODE_MISSING
```

Readiness codes should be added in the readiness slice, not mixed into the first schema implementation unless a test needs them.

## Effective Input Resolution

```text
effective value =
  incoming edge value
  else WorkflowNode.params[slot], only if dynamic=false
  else InputSlotDef.default_value
  else missing
```

If a slot is `dynamic=true`, `WorkflowNode.params[slot]` is structurally invalid and is not considered.

## External Readiness Capabilities

Core owns readiness traversal, workflow input context, `ModelRef`, and diagnostic target shapes. It does not own model manifests, model scanning, model descriptor resolution, file-system preflight, backend capability checks, or async I/O.

Readiness consumes host-provided external readiness providers. V1 keeps this provider synchronous from core's point of view. Hosts may call async services such as model-manager before invoking core readiness, then pass a snapshot-backed provider into core.

```text
ExternalReadinessProvider
  diagnostics_for(context, subject) -> Vec<Diagnostic>

ExternalReadinessContext
  workflow_id
  workflow_version
  node_id optional
  slot_id optional
  workflow_input_id optional
  path

ExternalReadinessSubject
  ModelRef
```

V1 uses `ExternalReadinessSubject::ModelRef` for `ParamValue::ModelRef`. The subject enum may grow later for path, image, workflow input, artifact destination, or backend capability checks.

Provider implementations do not live in core when they depend on concrete upstream crates. For example, a runtime or app-services adapter can depend on both core and model-manager, build a `ModelReadinessSnapshot`, and implement core's provider trait by looking up prepared results. Core must not depend on model-manager.

If core readiness encounters a subject that requires external readiness and the provider has no entry for it, readiness reports an error such as:

```text
CORE/WORKFLOW_EXTERNAL_READINESS_MISSING
```

This represents orchestration failure, not "model not found". A missing model should be represented by a normal model-manager diagnostic entry, such as `MODEL_MANAGER/MODEL_REF_NOT_FOUND`.

## Diagnostic Projection

Concrete external modules emit diagnostics about their own domains. Core readiness needs diagnostics attached to workflow inputs, nodes, or params. Projection turns an external diagnostic into a workflow-targeted diagnostic while preserving the original cause.

Generic projection can live in core because it depends only on core diagnostic types:

```text
project_external_diagnostic(original, new_primary)
  primary = new_primary
  related += original.primary
  keep id or derive a stable projected id
  keep correlation_id
  keep trace_span_id
  keep code
  keep severity
  keep source
  keep message
  keep fixes
```

For a model-manager diagnostic on a checkpoint param, the projected diagnostic should look like:

```text
primary:
  domain = workflow
  id = workflow_id
  path = nodes.node_checkpoint.params.checkpoint

related:
  target = original model-manager target
  message = external readiness source

code = MODEL_MANAGER/MODEL_SOURCE_STALE
source = model-manager
severity = Error
```

Warnings from external providers do not block plan construction. Error diagnostics block plan construction.

## Diagnostics

Diagnostics use an abstract target model.

```text
Diagnostic
  id: DiagnosticId
  correlation_id
  trace_span_id optional
  code
  severity
  source
  message
  primary: DiagnosticTarget
  related: Vec<DiagnosticRelated>
  fixes: Vec<DiagnosticFixHint>
```

```text
DiagnosticTarget
  domain
  id
  path
```

```text
DiagnosticRelated
  target: DiagnosticTarget
  message
```

```text
DiagnosticFixHint
  label
  description optional
  requires_confirmation
```

Examples:

```text
MissingRequiredInput
  target.domain = workflow.node
  target.id = node_sampler
  target.path = input_slots.model

InvalidParamForDynamicSlot
  target.domain = workflow.node
  target.id = node_sampler
  target.path = params.model

EdgeTargetSlotMissing
  target.domain = workflow.edge
  target.id = edge_x
  target.path = to.slot

ModelNotFound
  target.domain = workflow.node
  target.id = node_checkpoint
  target.path = params.checkpoint
```

Diagnostics are user-facing and agent-facing. Tracing/logging is developer-facing. They are linked by `correlation_id` and optional trace span IDs.

## Errors and Diagnostics

Errors and diagnostics are related but not identical.

```text
Error
  operation failed or could not complete normally

Diagnostic
  user/agent-facing explanation attached to a target

DomainEvent
  timeline/notification envelope that may carry diagnostics
```

Core should define a small trait boundary for service errors that can be surfaced to users, but it should not define a global error enum:

```text
DiagnosticSource
  diagnostic_source(&self) -> &'static str

DiagnosticError
  user_message(&self) -> String
  diagnostic_code(&self) -> DiagnosticCode
  diagnostic_severity(&self) -> DiagnosticSeverity

IntoDiagnostic
  into_diagnostic(id, target, correlation_id) -> Diagnostic
```

`IntoDiagnostic` may be provided as a lightweight helper or blanket implementation for errors that implement the diagnostic traits. Only errors that explicitly implement this diagnostic bridge should become diagnostics. Core should not turn every `std::error::Error` into a user-facing diagnostic automatically.

Concrete crates own their own error enums, such as `ConfigError`, `ModelManagerError`, `RuntimeError`, or `AgentError`. Those enums may use a local error library such as `thiserror`. User-visible cases should implement the core diagnostic bridge. Infrastructure errors can remain ordinary errors when they are not meaningful workflow/model diagnostics.

Core should not introduce a global `ReimagineError` enum. That would either force core to know every service's variants or collapse useful service errors into opaque strings.

Guideline:

```text
source missing
  diagnostic, not fatal process error

manifest JSON parse failed
  error plus diagnostic

internal invariant broken
  error, diagnostic optional
```

## Event and Diagnostic Payloads

Core should define shared event and diagnostic payload shapes that are host-neutral.

Diagnostics remain the actionable payload for users and agents. Events provide timeline and notification semantics.

```text
DomainEvent
  id
  correlation_id
  source
  kind
  subject
  diagnostics: Vec<Diagnostic>
  created_at
```

Examples:

```text
WorkflowCommandApplied
WorkflowCommandRejected
ModelAdded
ModelMarkedStale
RunStarted
RunFailed
```

Hosts bridge domain events into their transport:

```text
Tauri
  DomainEvent -> app.emit(...)

Axum
  DomainEvent -> SSE/WebSocket
```

Model manager should use this shared stream rather than introducing a Tauri-specific event concept.

## Fix Hints

Fix hints preview possible repairs but never apply directly.

```text
DiagnosticFixHint
  label
  description optional
  requires_confirmation
```

This early diagnostic payload slice does not embed `CommandBatch` previews. A later workflow command slice may add command preview support after `CommandBatch` is stable. All command-backed fixes must eventually go through `WorkflowSession.apply_batch()`.
