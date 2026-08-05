import type { ArtifactMetadata } from "@/ipc/schemas";

/**
 * Artifact display helpers (F5-4).
 *
 * `resolveArtifact` returns metadata whose `path` is an opaque string:
 *   - in mock mode it is a `data:` URL (or a dev HTTP URL) — usable as-is
 *   - in the desktop app it is a filesystem path — must be converted to
 *     a Tauri asset URL before an <img> can load it
 */

/** True when the current webview exposes the Tauri IPC bridge. */
export function tauriAvailable(): boolean {
  try {
    return (
      typeof window !== "undefined" &&
      typeof (window as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__ !==
        "undefined"
    );
  } catch {
    return false;
  }
}

/** Resolve an artifact metadata to an <img>-loadable URL. */
export async function artifactDisplayUrl(
  metadata: ArtifactMetadata,
): Promise<string> {
  const { path } = metadata;
  if (
    path.startsWith("data:") ||
    path.startsWith("http:") ||
    path.startsWith("https:") ||
    path.startsWith("asset:")
  ) {
    return path;
  }
  // Lazy import keeps the plain-browser bundle free of the Tauri chunk
  // (mirrors the dispatch pattern in ipc/commands.ts).
  if (tauriAvailable()) {
    const { convertFileSrc } = await import("@tauri-apps/api/core");
    return convertFileSrc(path);
  }
  return path;
}
