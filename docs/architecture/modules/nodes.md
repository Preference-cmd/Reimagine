# Nodes Module Architecture

> Status: working draft
> Crate: `crates/nodes`

## Role

`nodes` provides the V1 built-in node catalog, static registry, execution capability metadata, and external aliases. It uses `core` node-definition schemas and must not define a competing schema.

## Responsibilities

- Built-in `NodeDef` catalog.
- Static `NodeRegistry`.
- Execution capabilities.
- ComfyUI aliases.
- Node-local definition validation.

## Non-Responsibilities

- Canonical workflow storage.
- Core graph validation.
- ComfyUI parsing.
- Tauri IPC.
- Candle internals.
- Agent reasoning.

## Suggested Module Layout

```text
src/
  lib.rs
  def.rs
  registry.rs
  builtins.rs
  builtins/
    inputs.rs
    model.rs
    conditioning.rs
    latent.rs
    sampling.rs
    image.rs
  aliases.rs
  aliases/
    comfy.rs
```

## V1 Built-In SDXL Nodes

V1 should start with a small built-in catalog sufficient to express the SDXL base workflow example. These definitions live in `crates/nodes` but use the `core` node definition schema.

### Input and Utility

```text
builtin.string
  effect: Pure
  input_slots:
    value: String, dynamic=false, required=true
  output_slots:
    value: String, required=true
```

### Model Loading

```text
builtin.checkpoint_loader
  effect: Pure
  input_slots:
    checkpoint: ModelRef, dynamic=false, required=true
  output_slots:
    model: Model, required=true
    clip: Clip, required=true
    vae: Vae, required=true
```

The node receives a `ModelRef` saved in workflow params. Its runtime executor uses app-host-injected model resolution and backend loading capabilities to produce runtime handles.

### Conditioning

```text
builtin.clip_text_encode
  effect: Pure
  input_slots:
    clip: Clip, dynamic=true, required=true
    text: String, dynamic=true, required=true
  output_slots:
    conditioning: Conditioning, required=true
```

There is no separate SDXL-specific prompt encode node in V1. Encoding behavior is selected by the node executor/backend based on the loaded `Clip` handle and model metadata.

### Latent

```text
builtin.empty_latent_image
  effect: Pure
  input_slots:
    width: Integer, dynamic=false, required=true
    height: Integer, dynamic=false, required=true
    batch_size: Integer, dynamic=false, required=true
  output_slots:
    latent: Latent, required=true
```

### Sampling

```text
builtin.ksampler
  effect: Pure
  input_slots:
    model: Model, dynamic=true, required=true
    positive: Conditioning, dynamic=true, required=true
    negative: Conditioning, dynamic=true, required=true
    latent: Latent, dynamic=true, required=true
    seed: Seed, dynamic=false, required=true
    steps: Integer, dynamic=false, required=true
    cfg: Float, dynamic=false, required=true
    sampler: Select, dynamic=false, required=true
    scheduler: Select, dynamic=false, required=true
    denoise: Float, dynamic=false, required=true
  output_slots:
    latent: Latent, required=true
```

### Image

```text
builtin.vae_decode
  effect: Pure
  input_slots:
    vae: Vae, dynamic=true, required=true
    latent: Latent, dynamic=true, required=true
  output_slots:
    image: Image, required=true
```

```text
builtin.save_image
  effect: SideEffect
  input_slots:
    image: Image, dynamic=true, required=true
    filename_prefix: String, dynamic=false, required=true
  output_slots: []
```

`SaveImage` is runnable when required inputs resolve even though it has no outputs.

## ComfyUI Alias Notes

V1 aliases should be import-only metadata. They should not leak ComfyUI names into canonical workflow `type_id` values.

Initial aliases:

```text
CLIPTextEncode -> builtin.clip_text_encode
CheckpointLoaderSimple -> builtin.checkpoint_loader
EmptyLatentImage -> builtin.empty_latent_image
KSampler -> builtin.ksampler
VAEDecode -> builtin.vae_decode
SaveImage -> builtin.save_image
```
