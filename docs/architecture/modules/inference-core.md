# Inference Core Migration Note

> Status: folded into [`inference`](inference.md)
> Crate: `crates/inference-core` (temporary implementation detail)

## Role

`inference-core` was introduced as the low-level inference contract crate:
execution values, backend-affine handles, typed capability DTOs, router traits,
backend adapter traits, bridge policy, and inference errors lived there.

The architecture has since folded that contract layer back into the unified
[`inference`](inference.md) module. Future architecture and issue work should
treat `inference` as the owner of:

- built-in node orchestration;
- runtime-facing executor contracts;
- execution values and backend-affine handles;
- typed backend capability request/response DTOs;
- `InferenceRuntime` router contracts;
- `InferenceBackend` adapter contracts;
- backend selection and bridge policy contracts;
- model resolver handoff contracts;
- backend resource mechanism contracts;
- inference diagnostics and errors.

## Migration Rule

The physical `crates/inference-core` crate may remain while code migration is
split into small issues. During that migration, concrete code can still import
from `reimagine_inference_core` and re-export through `reimagine_inference`.

That crate is no longer a separate architecture module for new design work.
New issues should be filed under `inference/issues`, not
`inference-core/issues`, unless the task is explicitly about deleting or
mechanically migrating the remaining physical crate.

## Dependency Intent

The target architecture is:

```text
runtime -> inference
app-host -> inference
inference-backends/* -> inference

inference -> core

inference must not -> runtime
inference must not -> app-host
inference must not -> inference-backends/*
```

Concrete backend crates implement inference-owned contracts. Runtime consumes
the inference facade. App-host composes concrete backends, model resolver
adapters, router configuration, and executor registration.

## Historical Links

Historical issues may still reference `inference-core/01` and
`inference-core/02`; those are completed implementation steps and should not be
retconned. Future router/resource contract work is tracked under:

- `inference/issues/03-backend-resource-mechanism-contract.md`
- `inference/issues/04-configurable-backend-selection-policy.md`
