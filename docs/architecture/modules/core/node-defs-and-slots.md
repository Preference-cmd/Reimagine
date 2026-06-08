# Core Node Definitions and Slots

> Status: working draft
> Owner: `crates/core`

## Role

`NodeDef` describes a node type. `WorkflowNode` describes a node instance. A workflow node references a node definition through `type_id`.

```text
NodeDef.type_id = builtin.clip_text_encode
WorkflowNode.id = node_positive_encode
WorkflowNode.type_id = builtin.clip_text_encode
```

Multiple workflow nodes may share the same `type_id`; each must have a unique `id`.

## Slot-Based Inputs

Node inputs are modeled as input slots. Params are not a separate connectability system; they are saved fallback values for input slots.

```text
NodeDef
  type_id
  display_name
  category
  effect: NodeEffect
  input_slots: Vec<InputSlotDef>
  output_slots: Vec<OutputSlotDef>
```

```text
InputSlotDef
  id: SlotId
  kind: SlotKind
  dynamic: bool
  required: bool
  default_value: Option<ParamValue>
  constraints
  ui
```

```text
OutputSlotDef
  id: SlotId
  kind: SlotKind
  required: bool
```

## Rust Data Boundary

`NodeDef`, `InputSlotDef`, `OutputSlotDef`, `SlotKind`, and `NodeEffect` belong to `crates/core`.

The `nodes` crate owns concrete built-in registrations, but it must consume this schema rather than defining another node-definition model.

The old split between node `inputs`, `outputs`, and `parameters` should be removed as implementation reaches this slice. Static params are represented as `dynamic=false` input slots with optional defaults and UI metadata.

Minimum public types:

```text
NodeDef
NodeEffect
InputSlotDef
OutputSlotDef
SlotKind
SlotConstraint
SlotUi
NodeCatalog
```

`NodeCatalog` is a read-only lookup interface used by validation and command application:

```text
NodeCatalog
  get(type_id: &NodeTypeId) -> Option<&NodeDef>
```

The first implementation may use a simple in-memory catalog in tests. Built-in node registration belongs to `crates/nodes`.

## Slot Kinds

V1 slot kinds must cover the SDXL base workflow and saved params:

```text
String
Text
Integer
Float
Bool
Seed
Select
Path
ModelRef
Model
Clip
Vae
Latent
Conditioning
Image
Artifact
Null
```

`SlotKind` is the compatibility language for both params and edges. `ParamValue` values must map to compatible saved-value slot kinds. Runtime-handle slot kinds such as `Model`, `Clip`, `Vae`, `Latent`, `Conditioning`, and `Image` can be produced or consumed through edges but must not be stored directly in `WorkflowNode.params`.

## Dynamic Inputs

`dynamic` controls whether an input slot can have a static param fallback.

```text
dynamic=true
  edge-only input
  shown in the card connector area
  cannot appear in WorkflowNode.params

dynamic=false
  editable param input
  may have a saved fallback value
  may still be connected by an edge
  edge overrides the saved param during execution
```

All input slots support edge input. `dynamic=false` does not mean static-only; it means static-or-edge.

## Required Inputs and Outputs

`required` participates in executable readiness:

```text
InputSlotDef.required
  this input must resolve to an effective value before the node runs

OutputSlotDef.required
  if the node runs, the executor must produce this output
```

Effective input resolution:

```text
1. incoming edge value
2. WorkflowNode.params[slot], only when dynamic=false
3. InputSlotDef.default_value
4. missing
```

## Node Effects

```text
NodeEffect
  Pure
  SideEffect
```

Pure nodes contribute through outputs. A pure node whose required outputs are not consumed does not need to be scheduled.
More precisely, a pure node is included in an execution plan when at least one output is consumed by downstream edges or exposed through a workflow output. If included, all required outputs must be produced by the executor.

Side-effect nodes can be terminal run targets and may have no outputs. `SaveImage` is side-effectful: once its required inputs resolve, it can run even if it has no output slots.

## KSampler Example

```json
{
  "type_id": "builtin.ksampler",
  "effect": "Pure",
  "input_slots": [
    { "id": "model", "kind": "Model", "dynamic": true, "required": true },
    { "id": "positive", "kind": "Conditioning", "dynamic": true, "required": true },
    { "id": "negative", "kind": "Conditioning", "dynamic": true, "required": true },
    { "id": "latent", "kind": "Latent", "dynamic": true, "required": true },
    { "id": "seed", "kind": "Seed", "dynamic": false, "required": true },
    { "id": "steps", "kind": "Integer", "dynamic": false, "required": true },
    { "id": "cfg", "kind": "Float", "dynamic": false, "required": true },
    { "id": "sampler", "kind": "Select", "dynamic": false, "required": true },
    { "id": "scheduler", "kind": "Select", "dynamic": false, "required": true },
    { "id": "denoise", "kind": "Float", "dynamic": false, "required": true }
  ],
  "output_slots": [
    { "id": "latent", "kind": "Latent", "required": true }
  ]
}
```

## ModelRef vs Model Handles

`ModelRef` and `Model` are different slot kinds.

```text
ModelRef
  configuration-time reference
  saved in workflow params
  resolved outside core by a model-management service

Model / Clip / Vae
  runtime handles
  produced by loader nodes
  consumed by inference nodes
  not saved in workflow JSON
```

Loader or selector nodes use `ModelRef` as an input:

```json
{
  "type_id": "builtin.checkpoint_loader",
  "effect": "Pure",
  "input_slots": [
    { "id": "checkpoint", "kind": "ModelRef", "dynamic": false, "required": true }
  ],
  "output_slots": [
    { "id": "model", "kind": "Model", "required": true },
    { "id": "clip", "kind": "Clip", "required": true },
    { "id": "vae", "kind": "Vae", "required": true }
  ]
}
```

Execution nodes consume runtime handles:

```text
KSampler.model = Model
CLIPTextEncode.clip = Clip
VAEDecode.vae = Vae
```

`KSampler` should not accept `ModelRef`; it should not be responsible for resolving, loading, caching, and executing a checkpoint.

## SaveImage Example

```json
{
  "type_id": "builtin.save_image",
  "effect": "SideEffect",
  "input_slots": [
    { "id": "image", "kind": "Image", "dynamic": true, "required": true },
    { "id": "filename_prefix", "kind": "String", "dynamic": false, "required": true }
  ],
  "output_slots": []
}
```
