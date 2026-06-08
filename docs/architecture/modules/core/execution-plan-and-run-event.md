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

The plan is derived from a structurally valid workflow plus a node catalog. It includes only the execution subgraph needed for selected run targets.

Core does not store resolved `RuntimeValue` payloads in the plan. Runtime resolves node inputs into `RuntimeValue` at execution time using the run value store.

## Run Targets

V1 run targets:

```text
SideEffect node
Workflow output
```

If no explicit target is provided, V1 may default to all side-effect nodes. A workflow with no side-effect nodes and no workflow outputs is not executable.

## Planning Rules

The planner should:

- trace upstream dependencies from selected targets;
- include pure nodes only when their outputs contribute to a target;
- allow terminal side-effect nodes with no outputs;
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
