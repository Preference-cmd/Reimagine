# AGENTS.md - Reimagine

> The agent entry point. `CLAUDE.md` is a symlink - edit here only.
> Machine-local workflow rules: `AGENTS.local.md`. Process records: `.agents/`.

**Reimagine** is a Tauri + Burn + React desktop app with a native AI agent that
shares the project canvas and its workflow DSL with the user. GPL-3.0-or-later;
a desktop application, not a published library.

## Layout

24 Cargo workspace members: 23 crates under `crates/` plus `src-tauri/`.
`Cargo.toml` `[workspace]` is the authority; one line per crate is in
[`README.md`](README.md#workspace-layout).

```text
src-tauri/      Tauri 2 shell and IPC; the only crate that uses Tauri
crates/         23 workspace crates (members: Cargo.toml; one line each: README)
ui/             React 19 + Vite 7 frontend (Bun)
assets/         static placeholders
examples/       sample workflow JSON (axum-host E2E input)
docs/           machine-local architecture notes, design and generated catalogs (untracked)
.github/        ci-tauri.yml, ci-workers*.yml; scripts/ holds the tracked gates
```

The sibling `../agent-stack-oss` is **Causa**, the open-source publish set
(`causa` + `causa-{kernel,protocol,runtime,provider,extension}`). Root
`Cargo.toml` pins two legacy kernel crates to a Causa commit until the legacy
agent stack is removed.

## Commands

This list mirrors CI - a change is ready when all of it passes. Job names refer
to `ci-tauri.yml`.

```bash
# Rust gates - "quality" job
cargo fmt --all -- --check
cargo check --workspace --all-targets --locked --exclude reimagine-inference-burn-worker
cargo clippy --workspace --all-targets --locked --exclude reimagine-inference-burn-worker -- -D warnings
cargo test --workspace --locked --exclude reimagine --exclude reimagine-inference-burn --exclude reimagine-inference-burn-worker
cargo test -p reimagine-inference-burn --lib --locked --no-default-features --features flex   # Linux: Flex CPU
cargo test -p reimagine-inference-burn --lib --locked                                         # macOS: WGPU Metal

# Frontend - "quality" job (lint/format/test are hook-only today)
cd ui && bun install --frozen-lockfile
cd ui && bun run build

# Guards - "guards" job
python3 .github/scripts/check-module-layout.py
bash .github/scripts/check-dependency-directions.sh   # skips while cargo metadata fails
cargo deny check                                      # deny.toml; skips while cargo metadata fails
python3 .github/scripts/check-commit-messages.py range origin/main HEAD

# Machine-local, not CI: docs/ is untracked, so the generated catalogs are not gated
node scripts/gen-tool-catalog.mjs
node scripts/gen-node-catalog.mjs

# Machine-local proposal tree, not CI
node .agents/scripts/verify-proposals.mjs
node .agents/scripts/new-proposal.mjs <class> <slug>   # scaffold proposal + review record

# Entry points, not gates
cd src-tauri && cargo tauri dev
cd src-tauri && cargo tauri build
cd src-tauri && cargo tauri build --no-bundle   # what "compile" proves on macOS, Linux, Windows
cargo build -p reimagine-agent-daemon
cargo test  -p reimagine-agent-daemon
```

Git hooks: `lefthook install` once per clone (`lefthook.yml`); pre-commit runs
fmt, the layout guard and the whitespace check, commit-msg runs the message guard,
pre-push runs the check and the layering guard. The proposal verifier is
machine-local, so it runs from the gitignored `lefthook-local.yml` rather than
from a tracked job. Hooks report early; the list above, run by CI, is the gate.

The workspace resolves through a temporary git pin of the two legacy kernel
crates to a Causa commit. This allows the workspace and CI to resolve without
the sibling checkout. The pin is removed together with the legacy agent stack.

Toolchain: MSRV **1.96**, edition 2024. Do not raise `rust-version` or add a
dependency without a reason stated in the commit. Tests run offline; the
HuggingFace download path and the worker hardware smoke runs have their own
workflows.

## Conventions

- Module layout: no `mod.rs` under a crate's `src/` tree (test/bench/example
  fixtures may use `tests/common/mod.rs`); one file is `foo.rs`, two or more
  become a thin `foo.rs` index plus `foo/*.rs`; `#[path]` is rejected.
- Commits follow Conventional Commits (`feat`, `fix`, `refactor`, `perf`,
  `docs`, `test`, `chore`, `ci`, `build`, optional scope); a breaking commit
  updates `CHANGELOG.md` in the same push.
- Shipped comments and docstrings are self-contained: state the invariant, never
  cite `docs/` or `.agents/`.
- `Cargo.lock` is committed and is what ships, so an advisory against a locked
  version reaches users: update the lock, or add the advisory to `deny.toml`
  `ignore` with the reason it does not apply.
- License is GPL-3.0-or-later in `[workspace.package]` and `ui/package.json`;
  every crate inherits `publish = false`. A dependency's license must be
  GPL-compatible and on the `deny.toml` allow-list; no per-crate license files.
- `docs/` is untracked working memory: never cite it from shipped code, and
  never hand-edit `docs/generated/` - regenerate with `scripts/gen-*-catalog.mjs`.
- Dependency versions are centralized in root `Cargo.toml` `[workspace.dependencies]`.
- Burn is the primary inference backend; Candle is deprecated (V1 fallback only).
  Backend code lives in `crates/inference-backends/`.
- Prefer existing crate/module boundaries; keep `src-tauri` thin. Agent tools
  live in `crates/app-host/src/tools.rs`, not `agent-daemon`.
- When a crate, gate or workflow changes, update the copies together: Layout
  and Commands here, `README.md`, and the owning guard.
- Do not commit build outputs, local runtime data, model weights, secrets or
  machine-local planning files.

## Where to read more

| need | read |
|---|---|
| architecture and per-crate detail | `docs/architecture/overview.md` -> `docs/architecture/modules/` |
| product framing, crate one-liners | `README.md` |
| workflow and proposal lifecycle | `AGENTS.local.md` -> `.agents/pipelines/WORKFLOW.md` |
| design language and tokens | `docs/design/editor/DESIGN.md`, `ui/src/design/tokens.ts` |
| CI jobs and gate implementations | `.github/workflows/`, `.github/scripts/` |
| release notes and breaks | `CHANGELOG.md` |
| publish set, its names and version | the sibling `../agent-stack-oss` |
