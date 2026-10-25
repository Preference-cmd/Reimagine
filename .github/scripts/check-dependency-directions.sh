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
#   modules    aigc (inference composition, workers, model services)
#   backends   inference-burn, inference-burn-worker
#   workers    backend-worker-host, -transport-quic, -transport-grpc
#   host       app-host
#   entries    reimagine (src-tauri)
#
# In kernel-only builds (--no-default-features), app-host and reimagine
# must not reach the optional aigc module.
#
# No workspace crate depends on an agent stack (reimagine-agent-*, the legacy
# agent kernel crates, causa*). The operation pipeline in app-host serves
# humans and agents alike; an agent integration is an operator adapter above
# it, and the change that adds one grants that crate the only exception.
#
# tonic 0.14's gRPC server (the worker contract's server) depends on axum
# internally. The HTTP_SERVER ban still guards against direct HTTP-server
# usage: the worker crates' checks strip tonic's subtree from the tree, so
# an axum edge reachable any other way still fails.
#
# Patterns are extended regexes matched against cargo-tree lines after the
# "-- " edge marker, so the crate itself (the root line) never matches.
# End a crate name with " v" to stop it matching longer names:
# "reimagine-inference v" does not match "reimagine-inference-burn v".
set -euo pipefail

WORKSPACE='(reimagine(-[a-z-]+)?|aigc) v'
TAURI='tauri(-[a-z-]+)? v'
AGENT='(reimagine-agent-[a-z-]+|reimagine-ai-protocol|reimagine-context-kernel|causa(-[a-z]+)?) v'
APP_HOST='reimagine-app-host v'
ENTRIES='reimagine v'
BACKENDS='(reimagine-inference-(burn|burn-worker)|burn-[a-z]+|cubecl(-[a-z]+)?|wgpu) v'
HTTP_SERVER='axum v'
DOMAIN='(reimagine-core|reimagine-config|reimagine-nodes|reimagine-model-manager|reimagine-model-acquisition|reimagine-inference|reimagine-runtime|reimagine-plugin|reimagine-backend-worker-host|aigc) v'

tree_of() {
  cargo tree -p "$1" --edges normal --all-features --target all --locked --charset ascii
}

# Strip a named package's subtree from ASCII `cargo tree` output.
without_subtree() { # ... | without_subtree PACKAGE  (tree text on stdin)
  awk -v root=" $1 v" '
    function depth(line) {
      n = 0
      while (substr(line, 1, 4) == "|   " || substr(line, 1, 4) == "    ") {
        n++
        line = substr(line, 5)
      }
      return n
    }
    BEGIN { skipping = -1 }
    {
      if (skipping >= 0) {
        if (depth($0) > skipping) next
        skipping = -1
      }
      if (index($0, root) > 0) {
        skipping = depth($0)
        next
      }
      print
    }
  '
}

# check_without_tonic CRATE PATTERN... — like check, but the tree is
# matched with tonic's subtree removed (see the HTTP_SERVER note above).
check_without_tonic() {
  local root="$1"
  shift
  local tree found
  tree=$(tree_of "$root" | without_subtree tonic)
  local banned
  for banned in "$@"; do
    found=$(grep -oE -- "-- ${banned}" <<<"$tree" | head -n 1 || true)
    if [[ -n "$found" ]]; then
      echo "FAIL: $root must not depend on ${found#-- } (outside tonic)" >&2
      exit 1
    fi
  done
  echo "ok: $root"
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
# free of HTTP and the desktop shell (tokio sync-only is used for CancelToken).
check reimagine-core "$WORKSPACE" 'reqwest v' "$HTTP_SERVER" "$TAURI"
check reimagine-plugin "$WORKSPACE"
check reimagine-backend-worker-protocol "$WORKSPACE"

# Domain crates and contracts never reach a concrete inference backend,
# a host or the desktop shell. The runtime additionally
# stays out of model management, and the inference contract stays
# runtime-neutral.
for crate in reimagine-config reimagine-nodes reimagine-model-manager reimagine-model-acquisition; do
  check "$crate" "$BACKENDS" "$HTTP_SERVER"
done
check reimagine-inference "$BACKENDS" "$HTTP_SERVER" 'reimagine-runtime v'
check reimagine-runtime "$BACKENDS" "$HTTP_SERVER" 'reimagine-model-manager v'

# Backends implement the inference contract and know nothing above it.
# The worker binary reaches tonic's gRPC transport (the worker contract),
# which carries axum internally — see the HTTP_SERVER note above.
for crate in reimagine-inference-burn reimagine-inference-burn-worker; do
  check_without_tonic "$crate" "$HTTP_SERVER" 'reimagine-runtime v' 'reimagine-model-manager v'
done

# The worker host manages out-of-process workers and never links a backend;
# transports carry protocol messages and know no domain crate.
check_without_tonic reimagine-backend-worker-host "$BACKENDS" "$HTTP_SERVER"
for crate in reimagine-backend-worker-transport-quic reimagine-backend-worker-transport-grpc; do
  check "$crate" "$DOMAIN" "$BACKENDS"
done

# Every crate stays free of the agent stack. Only the desktop entry point
# uses Tauri; nothing below the entry points depends on the host facade;
# nothing depends on an entry point.
while read -r crate; do
  case "$crate" in
    reimagine) check "$crate" "$AGENT" "$ENTRIES" ;;
    *) check "$crate" "$AGENT" "$TAURI" "$APP_HOST" "$ENTRIES" ;;
  esac
done <<<"$members"

# Kernel-only assertion: app-host and entry crates compile without the
# optional aigc module when built with --no-default-features.
for crate in reimagine-app-host reimagine; do
  tree=$(cargo tree -p "$crate" --no-default-features --edges normal --target all --locked --charset ascii)
  if grep -E -- "aigc v" <<<"$tree" >/dev/null; then
    echo "FAIL: $crate --no-default-features must not depend on aigc" >&2
    exit 1
  fi
  echo "ok: $crate (kernel-only)"
done

echo "dependency directions OK"
