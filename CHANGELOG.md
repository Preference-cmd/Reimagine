# Changelog

All notable changes to Reimagine are recorded here. The format follows
Keep a Changelog; the version is the app version in `Cargo.toml`
`[workspace.package]` and in `src-tauri/tauri.conf.json`.

A breaking commit (`!` before the colon, or a `BREAKING CHANGE:` footer) must
add its note here in the same push - the commit-msg hook and the CI guards job
both check it.

## [Unreleased]

### Changed

- Workers speak protocol v2 over purpose-built gRPC calls (`Hello`,
  `Execute`, `Health`, `Cleanup`, `Shutdown` in `reimagine.worker.v2`)
  instead of a JSON message tunnel. The host reaches local workers over a
  private Unix socket (Windows: named pipe) with a per-launch bearer token;
  remote workers serve the same contract over HTTP/2 with TLS, a bearer
  token and TOFU certificate pinning, and advertise themselves over mDNS.
- Installed worker packages that only speak protocol v1 are rejected by the
  compatibility check with a reinstall diagnostic; old workers must be
  reinstalled.

- Document, run and download events arrive on one IPC subscription,
  `subscribe_events(channel)`, which replaces `subscribe_document_events`.
  Each event carries a `type` tag: `document`, `run` or `mod`.
- `run_workflow` no longer takes a channel; its run events arrive on
  `subscribe_events`, without replay, and their fields are camelCase (`runId`,
  `nodeId`, `artifact`, `createdAt`, ...).
- `download_huggingface_model` takes a caller-chosen `downloadId` instead of a
  channel; progress arrives as `aigc.download.{started,progress,completed,failed}`
  mod events whose payload is `downloadId`, `bytesDownloaded` (cumulative),
  `totalBytes` and `message`.
- The `run_workflow` response and its `initialSnapshot` use camelCase fields
  (`runId`, `workflowId`, `workflowVersion`, `initialSnapshot`, `nodeStates`,
  `startedAt`, ...); `workflowVersion` is a number.
- Board command results (`preview_board_commands`, `apply_board_commands`,
  `undo_board`, `redo_board`) return `version` instead of `boardVersion`.
- Board command results (`preview_board_commands`, `apply_board_commands`,
  `undo_board`, `redo_board`) no longer end their `changes` list with a
  `version_advanced` entry; the new version is the result's version field.
- `GET /health` and the `health` IPC command no longer return `status: "ok"`;
  they return `kernel` and `aigc` health objects (`state`, plus a `diagnostic`
  when unavailable).

### Removed

- The `reimagine-plugin` crate. Its plugin and extension descriptors had no
  caller; the plugin and extension ids moved to `reimagine-inference`, and the
  new `mods` crate defines the mod manifest, interface and registry.
- The `reimagine-axum-host` crate and its HTTP API (`cargo run -p reimagine-axum-host`).
  The desktop app (`src-tauri`) is now the only host adapter; humans and agents
  drive the kernel directly through `reimagine-app-host`.
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
- The unimplemented cross-worker tensor transfer and cross-backend bridge
  machinery: the `reimagine-backend-worker-transfer` crate (workspace members
  drop from 18 to 17), the `tensor_transfer_request`, `tensor_transfer_ack`
  and `tensor_data` messages of the worker wire protocol and their `.proto`
  mirrors, `aigc`'s `TopologyAwareBridgePolicy`, and `reimagine-inference`'s
  `BackendBridgePolicy` / `BridgePlan` / `RejectAllBridgePolicy` (the router
  constructor no longer takes a bridge policy; it rejects cross-backend
  handle conflicts inline, with the same behavior as the previous reject-all
  default). None of this was reachable in production: workers answered
  transfer messages with an "unexpected" diagnostic and the production
  router always used the reject-all policy.
- The QUIC worker transport and the standalone transport crates: the
  `reimagine-backend-worker-transport-quic` and
  `reimagine-backend-worker-transport-grpc` crates are gone (workspace
  members drop from 17 to 15); the gRPC contract, the local socket and TLS
  transports, the TOFU trust store, token handling and mDNS discovery now
  live in `reimagine-backend-worker-protocol`. Remote workers serve
  `https://` endpoints instead of `quic://`; remote workers were never
  routed, so no behavior is lost.

### Fixed

- The run panel stayed in "running" with no run id: the UI read camelCase
  fields from a `run_workflow` response that was snake_case.
- `cargo tauri dev` and `cargo tauri build` failed to start the frontend:
  `beforeDevCommand`/`beforeBuildCommand` ran `bun --cwd ../ui` from the
  repository root.
- Model downloads reported no files and 0 bytes: the report listed the
  staging directory after it had been moved into place. Files in
  subdirectories are now listed too.
- Download progress counted a file once per progress update, so the byte
  count overshot; it now tracks each file's latest transferred bytes.

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
