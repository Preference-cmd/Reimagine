# Core Module Architecture

> Status: working draft
> Crate: `crates/core`

## Role

`core` is the pure domain kernel. It defines the canonical workflow data model, node definition schema, command application semantics, validation and diagnostics, history, execution planning schema, run event schema, and backend-neutral shared values.

It must not depend on Tauri, Axum, Candle internals, React Flow, Rig, provider SDKs, or external workflow formats.

## Responsibilities

- Canonical workflow schema and single-file workflow JSON semantics.
- Shared model facade under `core::model`.
- Node definition schema, including slot-based inputs and outputs.
- `WorkflowCommand`, `CommandBatch`, `WorkflowChange`, `CommandResult`.
- `WorkflowSession` command application, versioning, history, undo, and redo.
- Structural validation and executable readiness diagnostics.
- Shared diagnostic model and diagnostic targets.
- `ExecutionPlan` and `RunEvent` schemas.
- Public execution values shared across runtime, inference, and backends.

## Non-Responsibilities

- Node catalog implementation and builtin node registration.
- Runtime scheduling, cancellation, or value store ownership.
- Tauri IPC, Axum routes, or host event emission.
- Candle model loading, tensors, devices, or backend payload stores.
- Agent provider calls or agent reasoning.
- ComfyUI parsing and import mechanics.
- UI projection state or React Flow DTOs.

## Document Split

Detailed core design is split by domain:

- [Workflow Schema](core/workflow.md)
- [Node Definitions and Slots](core/node-defs-and-slots.md)
- [Validation and Diagnostics](core/validation-and-diagnostics.md)
- [Commands and History](core/commands-and-history.md)
- [Execution Plan and Run Event](core/execution-plan-and-run-event.md)

The SDXL base workflow example is kept separately:

- [SDXL Base Workflow Example](../examples/sdxl-base-workflow.json)

## Suggested Module Layout

```text
src/
  lib.rs
  model.rs
  model/
    ids.rs
    values.rs
    models.rs
    artifacts.rs
    nodes.rs
    slots.rs
    workflow.rs
  workflow.rs
  workflow/
    node.rs
    edge.rs
    endpoint.rs
    interface.rs
    layout.rs
    metadata.rs
  node_def.rs
  command.rs
  command/
    batch.rs
    envelope.rs
    result.rs
    change.rs
  session.rs
  history.rs
  actor.rs
  validation.rs
  readiness.rs
  readiness/
    planner.rs
    targets.rs
    inputs.rs
    external.rs
    diagnostics.rs
  diagnostic.rs
  diagnostic/
    projection.rs
  proposal.rs
  execution_plan.rs
  run_event.rs
  execution_value.rs
  execution_value/
    value.rs
    handles.rs
    conditioning.rs
    tensor.rs
    backend.rs
```

Use modern Rust module layout. Do not introduce `mod.rs` files, and prefer ordinary `mod foo;` declarations over `#[path = "..."]` attributes.

## Shared Model Facade

`core::model` is the stable public facade for reusable data models and enums:

```rust
use reimagine_core::model::{
    ModelId, ModelRef, NodeId, ParamValue, TensorData, TensorDType, TensorShape,
};
```

Internal submodules should remain private and be re-exported from `model.rs`. External users should not depend on paths like `model::values::TensorData`.

`core::model` owns lightweight, backend-neutral types only:

```text
IDs:
  WorkflowId
  NodeId
  EdgeId
  RunId
  ArtifactId
  DiagnosticId
  HistoryEntryId
  CommandBatchId
  ProposalId
  ModelId
  NodeTypeId
  SlotId
  WorkflowInputId
  WorkflowOutputId
  WorkflowVersion

Values:
  ParamValue
  NodeValue
  TensorData
  TensorDType
  TensorShape

Models:
  ModelRef
  ModelSeries
  ModelVariant
  ModelRole

Artifacts:
  ArtifactRef
```

`core::model` does not own agent tools, provider config, ComfyUI schema,
Candle tensors, host DTOs, or the run-time execution value implementation.

Runtime execution values live under a focused `core::execution_value` module
and are re-exported through stable facade paths. Do not put the implementation
inside `core::model`; `model` remains the saved/editor semantic model facade,
while `execution_value` is the run-time value envelope.

Recommended public imports:

```rust
use reimagine_core::{
    ExecutionValue,
    ExecutionConditioning,
};

use reimagine_core::model::{
    ModelId, ModelRef, NodeValue, ParamValue,
};
```

The important boundary is that execution values are core-owned public handles,
not runtime-owned store internals and not inference-owned backend contracts.

## Public Execution Values

`core` owns the public value envelope used during workflow execution:

```text
ExecutionValue
  Param
  Model
  Clip
  Vae
  Latent
  Conditioning
  Image
  Artifact
  Null
```

The canonical name is `ExecutionValue`. Existing code may temporarily expose
`RuntimeValue` as a compatibility alias during migration, but new architecture
and new issue text should use `ExecutionValue`.

These values are the stable cross-crate handle shape that runtime stores,
inference executors consume, and backends construct or reinterpret through
typed capability requests. They are not workflow file contents and they are not
backend-local tensors.

`ExecutionConditioning` is the concrete value carried by
`ExecutionValue::Conditioning`. It represents conditioning produced during a
run, such as text embeddings and optional pooled embeddings, using
backend-affine tensor handles. Existing code may temporarily expose the old
`RuntimeConditioning` name as a compatibility alias during migration. Its
current `metadata` field is considered part of the conditioning value and does
not need a separate public abstraction in V1.

Conditioning metadata may include spatial context such as width, height, crop,
and target size. It must stay limited to information that node orchestration or
backend compatibility checks can treat as public execution context. Model-family
private conditioning internals must remain in backend-owned payloads.

Suggested internal shape:

```text
execution_value.rs
  facade and public re-exports

execution_value/value.rs
  ExecutionValue enum
  ExecutionValueKind

execution_value/handles.rs
  RuntimeModelHandle
  RuntimeClipHandle
  RuntimeVaeHandle
  RuntimeLatent
  RuntimeImage
  BackendTensorHandle

execution_value/conditioning.rs
  ExecutionConditioning
  ConditioningMetadata

execution_value/tensor.rs
  BackendTensorMetadata
  TensorShape / dtype references from model facade where reusable

execution_value/backend.rs
  BackendKind
  BackendPayloadKey
  BackendDeviceLabel
```

`lib.rs` should re-export the public execution value types directly from
`reimagine_core::*` where they are part of the cross-crate execution contract.
The implementation submodules should stay private. Consumers should not need to
import paths such as `core::execution_value::handles::RuntimeLatent`.

## Key Decisions

- Workflow JSON stores graph structure, node params, model refs, layout, interface, and metadata.
- Workflow JSON stores `ModelRef.id`, not model file paths.
- Workflow JSON does not store runtime values, backend tensor handles, loaded model handles, run sessions, diagnostics snapshots, or intermediate tensors.
- `core` owns the node-definition schema language: `NodeDef`, `InputSlotDef`, `OutputSlotDef`, `SlotKind`, `NodeEffect`, and the read-only `NodeCatalog` interface.
- `core` does not own the built-in node catalog data. Built-in definitions live in `crates/nodes` and consume the core schema.
- Node inputs are slot-based. `WorkflowNode.params` stores fallback values for `dynamic=false` input slots.
- Edges always connect slots. If an edge and param both provide a value for an input slot, the edge wins.
- `dynamic=true` input slots are edge-only and cannot appear in `WorkflowNode.params`.
- Structural validation failures reject command batches. Readiness failures block execution only.
- `WorkflowCommand` is the canonical workflow mutation language for both human and agent edits.
- `CommandBatch` is atomic in V1.
- History is snapshot-backed with cursor-based undo/redo.
