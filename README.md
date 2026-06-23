# reimagine

A Tauri + Candle + React desktop app for node-based image generation workflows — a ComfyUI alternative.

## Stack

- **Shell**: Tauri 2 (Rust)
- **Compute**: Hugging Face Candle (Rust ML framework)
- **UI**: React 19 + Vite 7 (TypeScript)
- **Workspace**: cargo workspace + bun (frontend)

## Quick start

```bash
cd ui && bun install && cd ..
cd src-tauri && cargo tauri dev
```

## Layout

- `src-tauri/` — Tauri shell + IPC.
- `crates/core/` — workflow engine (pure Rust).
- `ui/` — React frontend.
- `assets/` — static resources.

See [AGENTS.md](AGENTS.md) for the full workspace map and agent conventions.

## License

Reimagine is licensed under GPL-3.0-or-later. See [LICENSE](LICENSE).
