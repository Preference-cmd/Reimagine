# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root, or
- **`CONTEXT-MAP.md`** at the repo root if it exists — it points at one `CONTEXT.md` per context. Read each one relevant to the topic.
- **`docs/architecture/overview.md`** — read this as the architecture entry point.
- **`docs/architecture/modules/`** — read the module document that touches the area you're about to work in.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront.

Architecture overview and module documents are the source of truth for current
decisions. Local issue files should link back to these documents instead of
duplicating architecture text.

## File structure

Single-context repo (most repos):

```
/
├── CONTEXT.md
├── docs/architecture/
│   ├── overview.md
│   └── modules/
│       ├── core.md
│       └── model-manager.md
└── src/
```

Multi-context repo (presence of `CONTEXT-MAP.md` at the root):

```
/
├── CONTEXT-MAP.md
├── docs/architecture/                  ← system-wide architecture
└── src/
    ├── ordering/
    │   ├── CONTEXT.md
    │   └── docs/architecture/          ← context-specific architecture
    └── billing/
        ├── CONTEXT.md
        └── docs/architecture/
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/grill-with-docs`).

## Flag architecture conflicts

If your output contradicts the architecture overview or a module architecture document, surface it explicitly rather than silently overriding:

> _Contradicts docs/architecture/modules/model-manager.md — but worth reopening because..._
