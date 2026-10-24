# Changelog

All notable changes to Reimagine are recorded here. The format follows
Keep a Changelog; the version is the app version in `Cargo.toml`
`[workspace.package]` and in `src-tauri/tauri.conf.json`.

A breaking commit (`!` before the colon, or a `BREAKING CHANGE:` footer) must
add its note here in the same push - the commit-msg hook and the CI guards job
both check it.

## [Unreleased]

### Changed

- Board command results (`preview_board_commands`, `apply_board_commands`,
  `undo_board`, `redo_board`) no longer end their `changes` list with a
  `version_advanced` entry; the new version is in `boardVersion`.
- `GET /health` and the `health` IPC command no longer return `status: "ok"`;
  they return `kernel` and `aigc` health objects (`state`, plus a `diagnostic`
  when unavailable).

### Removed

- The deprecated in-process Candle backend: the `reimagine-inference-candle`
  crate and its `import-sdxl` binary, the `candle` cargo feature of
  `reimagine-app-host`, `aigc`, `reimagine-axum-host` and `reimagine`, and the
  Candle option in Settings. Inference runs only in the out-of-process Burn
  worker.
- A settings file that names `backend: "candle"` now loads as Burn with a
  warning; `candle:*` entries in `selected_instance`, `priority_order` and
  `disabled_instances` are dropped with a warning, and `candle_device` is
  ignored.
- `rebootstrap_backend` with `"candle"` and the HTTP `POST /models/convert` or
  `POST /models/acquire` with `target_backend: "candle"` are rejected as
  unsupported.
- Models imported in the Candle split layout (`converted_layout` =
  `candle_example_split`) cannot run on Burn. Their files and manifest entries
  are kept; import them again with Burn.
- The desktop IPC commands `create_agent_session`, `agent_turn` and
  `list_agent_providers`, and their frontend wrappers. The legacy native-agent
  integration is being removed; human editing, saving, running and proposal
  approval (`approve_proposal`) are unaffected.
- The legacy agent integration in `reimagine-app-host`: the agent service and
  sessions, the agent tools, provider configuration (`agent-providers.json` is
  no longer read) and the agent error variants. Workflow proposals stay in the
  workflow service and record their proposer as a `CommandActor`.
- The experimental `reimagine-agent-daemon` sidecar crate.
- New projects no longer get an `agent_threads` directory; existing ones are
  left untouched.

### Security

- Locked webpki-roots 1.0.10 and webpki-root-certs 1.0.10 to clear RustSec
  advisories RUSTSEC-2026-0381 and RUSTSEC-2026-0383.
- Locked h2 0.4.20, quinn-proto 0.11.19 and rustls 0.23.46 (with
  rustls-webpki 0.103.16 and aws-lc-rs 1.18.2) to clear their RustSec
  advisories, and chacha20 0.10.2 in place of the yanked 0.10.1.
- Added a `cargo deny` policy (`deny.toml`) for advisories, licenses, bans
  and sources, run by the CI guards job.

## [0.1.0] - baseline

- Baseline entry: this file was introduced with the development-workflow
  alignment. Earlier history lives in `git log`; the app version is 0.1.0 and
  has not been released as a public bundle yet.
