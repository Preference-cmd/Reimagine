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

/**
 * URL of a preview file named by the host's
 * `reimagine-asset://preview/<key>/<relative path>` URI.
 *
 * The host always reports the custom-scheme form; the webview needs the same
 * platform base as `getAssetUrl`, so only the path after `preview/` is kept and
 * each segment is encoded on its own.
 */
export function getPreviewUrl(previewUri: string): string {
  const base = tauriAvailable()
    ? convertFileSrc("", "reimagine-asset")
    : "reimagine-asset://localhost/";
  const withoutScheme = previewUri.replace(/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//, "");
  const segments = withoutScheme.split("/").filter((segment) => segment.length > 0);
  const previewIndex = segments.indexOf("preview");
  const relative = previewIndex === -1 ? segments : segments.slice(previewIndex + 1);
  return `${base}preview/${relative.map(encodeURIComponent).join("/")}`;
}
