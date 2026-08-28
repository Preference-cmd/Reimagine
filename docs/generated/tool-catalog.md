# Tool Catalog — Generated

> Generated from `crates/app-host/src/tools.rs` on 2026-08-28. Do not edit by hand — run `node scripts/gen-tool-catalog.mjs`.
> Freshness-gated by `node scripts/gen-tool-catalog.mjs --check` (CI lane `static`).

| Tool | Risk | Modes | Capability | Description |
|------|------|-------|------------|-------------|
| `workflow.get` | Read | Agent, Build | `workflow.read` | Get the current snapshot of a workflow. |
| `workflow.preview_commands` | Read | Agent, Build | `workflow.write` | Preview a command batch against a workflow without mutating it. |
| `workflow.propose_commands` | Editor | Agent, Build | `workflow.write` | Preview commands and store a pending proposal without mutating the workflow. |
| `workflow.apply_commands` | Editor | Agent, Build | `workflow.write` | Apply a command batch to a workflow. In Agent mode, low-risk editor-only batches may be auto-applied after a successful preview. |
| `model.list` | Read | Agent, Build | `model.read` | List available models in the workspace. |
| `model.resolve_ref` | Read | Agent, Build | `model.read` | Resolve a model reference to readiness or descriptor information. |
| `diagnostics.for_workflow` | Read | Agent, Build | `workflow.read` | Return immediate diagnostics for the current workflow and session. |
| `model.download` | External | Agent, Build | `model.write` | Download a HuggingFace model into the workspace models directory. |

## Input schemas (first 120 chars)

- **workflow.get**: `workflow_id required: workflow_id`
- **workflow.preview_commands**: `workflow_id, batch required: workflow_id, batch`
- **workflow.propose_commands**: `workflow_id, proposal_id, batch, created_at required: workflow_id, proposal_id, batch, created_at`
- **workflow.apply_commands**: `workflow_id, batch required: workflow_id, batch`
- **model.list**: ``
- **model.resolve_ref**: `model_ref required: model_ref`
- **diagnostics.for_workflow**: `workflow_id required: workflow_id`
- **model.download**: `repo_id required: repo_id`

## Stats

- Total tools: 8
- By risk: Read=5, Editor=2, External=1
