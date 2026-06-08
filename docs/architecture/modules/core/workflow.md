# Core Workflow Schema

> Status: working draft
> Owner: `crates/core`

## Role

The workflow schema is the canonical saved representation of a Reimagine graph. It describes how to generate a result, not what happened during a run.

Workflow JSON stores:

- graph structure;
- node params as static fallback values;
- model references;
- workflow interface;
- layout;
- metadata.

Workflow JSON does not store:

- `RuntimeValue`;
- backend tensor handles;
- loaded model handles;
- Candle tensors;
- `RunValueStore`;
- `RunSession`;
- run events;
- diagnostics snapshots by default.

## Canonical Shape

```text
Workflow
  schema_version: WorkflowSchemaVersion
  id: WorkflowId
  version: WorkflowVersion
  metadata: WorkflowMetadata
  interface: WorkflowInterface
  nodes: Vec<WorkflowNode>
  edges: Vec<WorkflowEdge>
  layout: WorkflowLayout
```

```text
WorkflowNode
  id: NodeId
  type_id: NodeTypeId
  label: Option<String>
  params: Map<SlotId, ParamValue>
  metadata
```

```text
WorkflowEdge
  id: EdgeId
  from: Endpoint
  to: Endpoint
```

```text
Endpoint
  NodeSlot { node: NodeId, slot: SlotId }
  WorkflowInput { id: WorkflowInputId }
  WorkflowOutput { id: WorkflowOutputId }
```

V1 workflow examples can use this JSON form:

```json
{ "node": "node_sampler", "slot": "latent" }
```

`node` references `WorkflowNode.id`; it is not a separate identifier namespace.

## Rust Data Boundary

`core-workflow/01` should introduce the canonical workflow data types in `crates/core`, not in `nodes`, `runtime`, `src-tauri`, or the UI.

Minimum public types:

```text
Workflow
WorkflowSchemaVersion
WorkflowVersion
WorkflowMetadata
WorkflowInterface
WorkflowInputDef
WorkflowOutputDef
WorkflowNode
WorkflowEdge
Endpoint
WorkflowLayout
NodeLayout
Position
Viewport
```

Workflow ID-like fields should use the `core::model` facade:

```text
WorkflowId
NodeId
EdgeId
NodeTypeId
SlotId
WorkflowInputId
WorkflowOutputId
```

The implementation should keep internal modules private and re-export stable public types through `workflow.rs` and, where shared across domains, `core::model`.

## Serde Strategy

Workflow JSON is a stable user-facing format. Serde derives are part of the V1 contract.

Rules:

- `schema_version` is the literal string `reimagine.workflow.v1` for V1.
- `version` is a monotonically increasing integer stored in the workflow file.
- `nodes` and `edges` are stored as arrays to preserve file readability and authoring order.
- `params`, `layout.nodes`, and metadata extension maps should use deterministic map ordering in serialization.
- Unknown optional metadata should roundtrip only when explicitly modeled. V1 does not need an arbitrary JSON extension bag unless a concrete consumer needs it.
- Runtime-only state must not appear in workflow serde output.

The SDXL base example under `docs/architecture/examples/sdxl-base-workflow.json` is a contract test fixture. It must parse into `Workflow` and serialize back to the same canonical shape, allowing only deterministic formatting differences.

## Params

`WorkflowNode.params` stores fallback values for `dynamic=false` input slots.

```json
{
  "id": "node_sampler",
  "type_id": "builtin.ksampler",
  "params": {
    "steps": { "type": "integer", "value": 30 },
    "cfg": { "type": "float", "value": 7.0 }
  }
}
```

Params are not runtime values. They are saved editor/configuration values.

If an input slot has both a param and an incoming edge, the edge is the effective value during execution. The param remains saved as a fallback and becomes effective again if the edge is disconnected.

`ParamValue` should remain the saved/editor value enum. It may contain `ModelRef`, strings, numbers, booleans, seeds, selects, paths, and null-like values. It must not contain `RuntimeValue`, loaded handles, or Candle tensor payloads.

## Interface

Workflow-level inputs and outputs are optional in V1 but reserved in the schema.

```text
WorkflowInterface
  inputs: Vec<WorkflowInputDef>
  outputs: Vec<WorkflowOutputDef>
```

Workflow interface endpoints are distinct from node slots:

```json
{ "workflow_input": "positive_prompt" }
{ "workflow_output": "image" }
```

V1 desktop SDXL workflows may keep these arrays empty and rely on terminal side-effect nodes such as `SaveImage`.

## Layout

Layout is canonical workflow metadata:

```text
WorkflowLayout
  nodes: Map<NodeId, Position>
  viewport optional
```

Layout participates in save/load, versioning, history, undo/redo, provenance, and agent proposals. It does not affect execution semantics.

`core-workflow/01` only needs the saved layout data model and structural checks for dangling node layout. Command-level layout mutation and history integration belong to the command/session slice.

## Example

See [SDXL Base Workflow Example](../../examples/sdxl-base-workflow.json).
