#!/usr/bin/env bash
# Dependency-direction guard: enforce Reimagine's layering machine-side.
#
# Checks the *normal* dependency graph with all features enabled, on every
# target platform, so an optional or platform-only edge counts too.
# Dev-dependencies are exempt by design: tests may wire layers together,
# shipped code may not.
#
#   leaves     core, plugin, backend-worker-protocol    no workspace crate
#   domain     config, nodes, model-manager, model-acquisition,
#              inference, runtime
#   backends   inference-candle, inference-burn, inference-burn-worker
#   workers    backend-worker-host, -transport-quic, -transport-grpc, -transfer
#   agent      agent-harness, agent-legacy-stack, agent-extension,
#              agent-macros and the agent kernel crates they use
#   host       app-host
#   entries    reimagine (src-tauri), axum-host, agent-daemon
#
# Patterns are extended regexes matched against cargo-tree lines after the
# "-- " edge marker, so the crate itself (the root line) never matches.
# End a crate name with " v" to stop it matching longer names:
# "reimagine-inference v" does not match "reimagine-inference-burn v".
set -euo pipefail

WORKSPACE='reimagine(-[a-z-]+)? v'
TAURI='tauri(-[a-z-]+)? v'
AGENT='(reimagine-agent-[a-z-]+|reimagine-ai-protocol|reimagine-context-kernel|causa(-[a-z]+)?) v'
APP_HOST='reimagine-app-host v'
ENTRIES='(reimagine|reimagine-axum-host|reimagine-agent-daemon) v'
BACKENDS='(reimagine-inference-(burn|burn-worker|candle)|burn-[a-z]+|candle-[a-z]+|cubecl(-[a-z]+)?|wgpu) v'
HTTP_SERVER='axum v'
DOMAIN='(reimagine-core|reimagine-config|reimagine-nodes|reimagine-model-manager|reimagine-model-acquisition|reimagine-inference|reimagine-runtime|reimagine-plugin|reimagine-backend-worker-host) v'

tree_of() {
  cargo tree -p "$1" --edges normal --all-features --target all --locked --charset ascii
}

# check CRATE PATTERN... — CRATE's graph reaches none of PATTERN.
check() {
  local root="$1"
  shift
  local tree
  tree=$(tree_of "$root")
  local banned found
  for banned in "$@"; do
    found=$(grep -oE -- "-- ${banned}" <<<"$tree" | head -n 1 || true)
    if [[ -n "$found" ]]; then
      echo "FAIL: $root must not depend on ${found#-- }" >&2
      exit 1
    fi
  done
  echo "ok: $root"
}

members=$(cargo metadata --no-deps --format-version 1 --locked |
  python3 -c 'import json, sys; print("\n".join(sorted(p["name"] for p in json.load(sys.stdin)["packages"])))')
if [[ -z "$members" ]]; then
  echo "FAIL: cargo metadata listed no workspace members" >&2
  exit 1
fi

# Leaves: the domain vocabulary, the plugin descriptors and the worker wire
# protocol depend on no other workspace crate. The domain root also stays
# free of async runtimes, HTTP and the desktop shell.
check reimagine-core "$WORKSPACE" 'tokio v' 'reqwest v' "$HTTP_SERVER" "$TAURI"
check reimagine-plugin "$WORKSPACE"
check reimagine-backend-worker-protocol "$WORKSPACE"

# Domain crates and contracts never reach a concrete inference backend,
# the agent stack, a host or the desktop shell. The runtime additionally
# stays out of model management, and the inference contract stays
# runtime-neutral.
for crate in reimagine-config reimagine-nodes reimagine-model-manager reimagine-model-acquisition; do
  check "$crate" "$BACKENDS" "$AGENT" "$HTTP_SERVER"
done
check reimagine-inference "$BACKENDS" "$AGENT" "$HTTP_SERVER" 'reimagine-runtime v'
check reimagine-runtime "$BACKENDS" "$AGENT" "$HTTP_SERVER" 'reimagine-model-manager v'

# Backends implement the inference contract and know nothing above it.
for crate in reimagine-inference-candle reimagine-inference-burn reimagine-inference-burn-worker; do
  check "$crate" "$AGENT" "$HTTP_SERVER" 'reimagine-runtime v' 'reimagine-model-manager v'
done

# The worker host manages out-of-process workers and never links a backend;
# transports and transfer carry protocol messages and know no domain crate.
check reimagine-backend-worker-host "$BACKENDS" "$AGENT" "$HTTP_SERVER"
for crate in reimagine-backend-worker-transport-quic reimagine-backend-worker-transport-grpc reimagine-backend-worker-transfer; do
  check "$crate" "$DOMAIN" "$BACKENDS" "$AGENT"
done

# Only the desktop entry point uses Tauri; nothing below the entry points
# depends on the host facade; nothing depends on an entry point.
while read -r crate; do
  case "$crate" in
    reimagine) check "$crate" "$ENTRIES" ;;
    reimagine-axum-host | reimagine-agent-daemon) check "$crate" "$TAURI" "$ENTRIES" ;;
    *) check "$crate" "$TAURI" "$APP_HOST" "$ENTRIES" ;;
  esac
done <<<"$members"

echo "dependency directions OK"
