import { convertFileSrc } from "@tauri-apps/api/core";

/** True when the current webview exposes the Tauri IPC bridge. */
function tauriAvailable(): boolean {
  try {
    return (
      typeof window !== "undefined" &&
      typeof (window as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__ !== "undefined"
    );
  } catch {
    return false;
  }
}

/**
 * URL of a project asset served by the `reimagine-asset` protocol.
 *
 * `convertFileSrc` picks the platform's form (`reimagine-asset://localhost/`
 * or `http://reimagine-asset.localhost/` on Windows and Android) but
 * percent-encodes `/`, so it only supplies the base and each id is encoded
 * as its own path segment.
 */
export function getAssetUrl(projectId: string, assetId: string): string {
  const base = tauriAvailable()
    ? convertFileSrc("", "reimagine-asset")
    : "reimagine-asset://localhost/";
  return `${base}${encodeURIComponent(projectId)}/${encodeURIComponent(assetId)}`;
}
