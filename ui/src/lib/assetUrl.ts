import { convertFileSrc } from "@tauri-apps/api/core";
import { tauriAvailable } from "@/lib/artifacts";

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
