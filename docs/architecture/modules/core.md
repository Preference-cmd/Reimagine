# Core Module Architecture

> Status: working draft
> Crate: `crates/core`

## Role

`core` is the pure domain kernel. It defines the canonical workflow data model,
node definition schema, command application semantics, validation and
diagnostics, history, execution planning schema, run event schema, and
backend-neutral saved/editor values.

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

## Non-Responsibilities

- Node catalog implementation and builtin node registration.
- Runtime scheduling, cancellation, or value store ownership.
- Runtime execution values, backend-affine tensor/model handles, or loaded
  inference payload handles.
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
Candle tensors, host DTOs, or the runtime execution value implementation.
Execution values are internal runtime/inference data and live in the inference
execution stack, with `inference-core` owning the canonical low-level shape and
`inference` re-exporting the runtime-facing facade.

Recommended public imports:

```rust
use reimagine_core::model::{
    ModelId, ModelRef, NodeValue, ParamValue,
};
```

The important boundary is that `core` owns durable workflow semantics and
host-safe observation schemas, not transient execution handles. Workflow JSON,
snapshots, summaries, run events, Axum/Tauri DTOs, and Agent tool results must
not expose `ExecutionValue`, backend tensor handles, loaded model handles, or
other inference runtime internals.

## Key Decisions

- Workflow JSON stores graph structure, node params, model refs, layout, interface, and metadata.
- Workflow JSON stores `ModelRef.id`, not model file paths.
- Workflow JSON does not store runtime values, backend tensor handles, loaded model handles, run sessions, diagnostics snapshots, or intermediate tensors.
- `core` owns the node-definition schema language: `NodeDef`, `InputSlotDef`, `OutputSlotDef`, `SlotKind`, `NodeEffect`, and the read-only `NodeCatalog` interface.
- `core` does not own the built-in node catalog data. Built-in definitions live in `crates/nodes` and consume the core schema.
- `core` does not own `ExecutionValue`; canonical execution values and
  backend-affine handles live in `inference-core` and are re-exported by
  `inference` for runtime-facing code.
- Node inputs are slot-based. `WorkflowNode.params` stores fallback values for `dynamic=false` input slots.
- Edges always connect slots. If an edge and param both provide a value for an input slot, the edge wins.
- `dynamic=true` input slots are edge-only and cannot appear in `WorkflowNode.params`.
- Structural validation failures reject command batches. Readiness failures block execution only.
- `WorkflowCommand` is the canonical workflow mutation language for both human and agent edits.
- `CommandBatch` is atomic in V1.
- History is snapshot-backed with cursor-based undo/redo.
