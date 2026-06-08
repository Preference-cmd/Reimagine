# Core Commands and History

> Status: working draft
> Owner: `crates/core`

## Role

`WorkflowCommand` is the canonical workflow mutation language. Human edits, agent edits, imports, and future host actions all modify workflows by producing command batches.

Commands are not generic app actions. The following are not workflow commands:

- run workflow;
- cancel run;
- save or open workflow files;
- import ComfyUI workflow;
- start or send agent messages;
- list or rescan models.

Those actions may produce workflow commands, but they are not commands themselves.

## V1 Command Set

```text
WorkflowCommand
  AddNode
  RemoveNode
  Connect
  Disconnect
  SetParam
  RemoveParam
  MoveNode
  ApplyLayout
  SetNodeLabel
  SetWorkflowMetadata
```

`SetParam` and `RemoveParam` operate on `dynamic=false` input slots only. Dynamic input slots are edge-only.

## Command Batch

```text
CommandBatch
  id: CommandBatchId
  actor
  base_version
  provenance
  correlation_id
  commands: Vec<WorkflowCommand>
```

V1 command batches are atomic:

```text
all commands structurally validate and apply
or no commands apply
```

If `base_version` does not match the session version, the batch is rejected with a version conflict diagnostic.

## Apply Flow

```text
1. check base_version
2. clone workflow
3. apply commands to clone
4. run structural validation on clone
5. reject if structural errors exist
6. compute forward and inverse changes
7. commit clone
8. advance version
9. append history entry
10. run executable readiness
11. return CommandResult
```

Readiness diagnostics do not roll back a structurally valid command batch.

## Command Result

```text
CommandResult
  status
  workflow_version
  changes: Vec<WorkflowChange>
  diagnostics: Vec<Diagnostic>
  history_entry_id optional
```

```text
CommandStatus
  Applied
  Rejected
  NoOp
```

Rejected results have no changes and contain structural diagnostics. Applied results may still contain readiness diagnostics.

## Workflow Changes

`WorkflowChange` records actual state changes:

```text
WorkflowChange
  NodeAdded
  NodeRemoved
  EdgeAdded
  EdgeRemoved
  ParamSet
  ParamRemoved
  NodeMoved
  LayoutApplied
  NodeLabelSet
  WorkflowMetadataSet
  VersionAdvanced
```

Uses:

- UI patch projection;
- undo;
- redo;
- agent proposal diff;
- history details.

Setting a field to its current value should produce no change. A batch with no changes returns `NoOp`.

## Command Semantics

### AddNode

```text
AddNode
  node_id
  type_id
  label optional
  params
  position optional
```

Rules:

- node id must be unique;
- type id must resolve through the node catalog;
- params must reference `dynamic=false` input slots;
- param value kinds must match slot kinds;
- optional position writes canonical layout.

Missing required inputs are readiness diagnostics, not structural failures.

### RemoveNode

Removes the node, all connected edges, and node layout.

History inverse stores the node snapshot, removed edge snapshots, and layout snapshot.

### Connect

```text
Connect
  edge_id
  from: Endpoint
  to: Endpoint
```

Rules:

- edge id must be unique;
- from node and to node must exist;
- from slot must be an output slot;
- to slot must be an input slot;
- slot kinds must be compatible;
- each input slot has at most one incoming edge in V1.

Connecting to a `dynamic=false` input slot does not delete the saved param fallback. Edge value wins during execution.

### Disconnect

Removes one edge. If the disconnected target has a saved param fallback, that fallback becomes effective again.

### SetParam

Rules:

- node must exist;
- slot must be an input slot;
- slot must have `dynamic=false`;
- value kind must match slot kind.

If the slot currently has an incoming edge, the command is still allowed. It updates the fallback value but does not affect the current effective value.

### RemoveParam

Removes a saved fallback value from a `dynamic=false` input slot. If the slot has no incoming edge or default and is required, readiness reports a missing input diagnostic.

### MoveNode and ApplyLayout

Layout is canonical workflow metadata. Layout commands participate in versioning, history, provenance, proposals, save/load, and undo/redo.

### SetNodeLabel and SetWorkflowMetadata

Labels are per-node instance display names and may repeat. Empty labels are allowed; UI may fall back to the node definition display name.

V1 metadata command can replace metadata as a whole rather than implementing path-level patches.

## History

V1 uses snapshot-backed history.

```text
HistoryEntry
  id: HistoryEntryId
  actor
  provenance
  command_batch
  before: Workflow
  after: Workflow
  forward_changes: Vec<WorkflowChange>
  inverse_changes: Vec<WorkflowChange>
  created_at
```

`WorkflowHistory` uses a cursor:

```text
WorkflowHistory
  entries: Vec<HistoryEntry>
  cursor: usize
```

Undo/redo move the cursor, restore a snapshot, increment version, and emit changes. They do not append ordinary history entries.

## Agent Proposals

Agent build mode uses dry-run command batches:

```text
WorkflowProposal
  id: ProposalId
  base_version
  command_batch
  preview_changes
  diagnostics
```

Human accept applies the same batch through `WorkflowSession.apply_batch()`.
