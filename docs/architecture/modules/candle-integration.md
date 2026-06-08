# Candle Integration Module Architecture

> Status: working draft
> Crate: `crates/candle-integration`

## Role

`candle-integration` is the Candle-specific inference backend. It implements backend behavior behind the `core` inference contracts.

## V1 Target

V1 must support SDXL base-only text-to-image inference. SDXL refiner support is deferred.

## Responsibilities

- Session management.
- Device and dtype configuration.
- Model loading and cache.
- CLIP, UNet, VAE implementations.
- Tensor conversion between `core` data and Candle tensors.

## Non-Responsibilities

- Workflow graph semantics.
- Tauri IPC.
- Runtime scheduling.
- Agent tools.
- ComfyUI import.
