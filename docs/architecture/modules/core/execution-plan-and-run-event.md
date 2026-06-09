# Core Execution Plan and Run Event

> Status: working draft
> Owner: `crates/core`

## Role

`ExecutionPlan` is the host-independent handoff from core readiness into runtime execution. It describes what should run and in which dependency order. It does not execute nodes, own runtime values, load models, or schedule tasks.

`RunEvent` is the host-neutral event payload emitted by runtime and bridged by hosts such as Tauri or Axum.

## Execution Plan Boundary

Core owns the plan schema:

```text
ExecutionPlan
  workflow_id
  workflow_version
  target_selection
  targets: Vec<RunTarget>
  nodes: Vec<ExecutionNode>
  edges: Vec<ExecutionEdge>
  stages: Vec<ExecutionStage>
```

```text
ExecutionNode
  node_id
  type_id
  input_bindings
  output_slots
```

```text
ExecutionStage
  index
  node_ids: Vec<NodeId>
```

The plan is derived from a structurally valid workflow plus a node catalog. It includes only the execution subgraph needed for the selected run targets.

Core does not store resolved `RuntimeValue` payloads in the plan. Runtime resolves node inputs into `RuntimeValue` at execution time using the run value store.

## Plan Result

Readiness returns a plan result rather than emitting a stream:

```text
ExecutionPlanResult
  plan: Option<ExecutionPlan>
  report: OperationReport
```

If readiness diagnostics contain errors, `plan` is `None`. Warning diagnostics may be returned with `plan = Some(...)`.

`OperationReport` is the synchronous return envelope for diagnostics and domain events produced by one operation. It is not the event-bus message type. Runtime progress uses `RunEvent`, not `OperationReport`.

## Target Selection

Execution planning is always produced for a target selection, not for the whole workflow by default:

```text
RunTargetSelection
  AllDefaultTargets
  ExplicitTargets(Vec<RunTarget>)
```

`AllDefaultTargets` is the V1 "run graph" behavior. It selects every default target in workflow order and produces a single merged plan. Shared upstream nodes execute once even when multiple targets depend on them.

`ExplicitTargets` supports ComfyUI-style partial execution. A caller may run only a selected terminal node, a selected node output, or a workflow output.

```text
RunTarget
  Node { node_id }
  NodeOutput { node_id, slot_id }
  WorkflowOutput { output_id }
```

`RunTarget::Node` is valid only for target-capable nodes.

## Target-Capable Nodes

A node is target-capable when its `NodeDef` has no `required=true` output slots. This describes terminal nodes where executing the node itself is the desired result.

Examples:

- `SaveImage`
- `PreviewImage`
- future `ExportArtifact`
- future host/bridge output nodes

`NodeEffect::SideEffect` remains execution metadata, not target eligibility. A side-effect node is usually target-capable, but the default target rule is based on required output slots rather than effect.

A target-capable node is runnable only when all required effective inputs resolve. Missing required inputs are readiness diagnostics.

If `AllDefaultTargets` finds no target-capable nodes and the workflow exposes no default workflow outputs, readiness reports no run target.

## Planning Rules

The planner should:

- trace upstream dependencies from selected targets;
- merge multiple target traces into one execution subgraph;
- include pure nodes only when their outputs contribute to a target;
- allow target-capable terminal nodes with no required outputs;
- reject executable cycles through readiness diagnostics;
- produce stages where nodes in the same stage have no dependencies on each other;
- preserve deterministic node ordering within stages.

## Run Event

Core owns the shared `RunEvent` payload. Runtime owns `RunEventSink`.

Minimum V1 event kinds:

```text
RunQueued
RunStarted
RunCompleted
RunFailed
RunCancelled
NodeQueued
NodeStarted
NodeCompleted
NodeFailed
NodeSkipped
NodeCancelled
ArtifactCreated
PreviewUpdated
DiagnosticEmitted
```

`RunEvent` shape:

```text
RunEvent
  id
  run_id
  workflow_id
  workflow_version
  kind
  node_id optional
  artifact optional
  diagnostics: Vec<Diagnostic>
  created_at
  correlation_id
```

Run events are timeline payloads. Diagnostics remain the user-facing explanation. Tracing/logging remains developer-facing and links through `correlation_id` or trace span IDs.

## Implementation Slicing

`core-workflow/03` should introduce `ExecutionPlan`, readiness diagnostics, and `RunEvent` schemas. It should not implement runtime scheduling, cancellation, backend model loading, or artifact writing.

Suggested implementation split:

```text
execution_plan.rs
  plan/result/target/stage data shapes

run_event.rs
  RunEvent and RunEventKind data shapes

readiness.rs
  public readiness API and re-exports

readiness/
  targets.rs      target selection and target validation
  inputs.rs       effective input resolution
  planner.rs      upstream tracing, cycle checks, and stage planning
  external.rs     ExternalReadinessProvider and subjects
  diagnostics.rs  readiness diagnostic constructors

diagnostic/projection.rs
  generic diagnostic projection helper
```

Keep core readiness synchronous and host-neutral. Concrete adapters that depend on model-manager, runtime, or filesystem state belong outside core.
