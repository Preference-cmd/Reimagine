import { create } from "zustand";
import type { RunEventPayload } from "@/ipc/schemas";
import { resolveArtifact } from "@/ipc";
import { artifactDisplayUrl } from "@/lib/artifacts";

/**
 * Artifact preview cache (F5-4).
 *
 * Consumes run events forwarded from the runtime store and resolves
 * artifact ids to displayable URLs via `resolveArtifact()` IPC, keyed by
 * node id. A new run clears the map; pending entries are marked stale on
 * terminal run states so no spinner outlives the run.
 *
 * Nodes render previews through the `useNodeArtifact` hook — this store
 * holds no React state itself.
 */

export type ArtifactPreviewStatus = "pending" | "ready" | "error" | "stale";

export type ArtifactPreview = {
  artifactId: string;
  runId: string;
  nodeId: string;
  status: ArtifactPreviewStatus;
  /** <img>-loadable URL (mock data: URL, dev http URL, or Tauri asset URL). */
  url?: string;
  error?: string;
};

type ArtifactState = {
  runId: string | null;
  /** nodeId → latest preview for the current run. */
  byNode: Record<string, ArtifactPreview>;
  /** Feed a run event (called by the runtime store for every event). */
  handleEvent: (event: RunEventPayload) => void;
  /** Start a fresh run — clears all previews. */
  beginRun: (runId: string) => void;
  reset: () => void;
};

const TERMINAL_KINDS = new Set(["RunCompleted", "RunFailed", "RunCancelled"]);
const ARTIFACT_KINDS = new Set(["ArtifactCreated", "PreviewUpdated"]);

export const useArtifactStore = create<ArtifactState>()((set, get) => ({
  runId: null,
  byNode: {},

  beginRun: (runId: string) => {
    if (get().runId === runId) return;
    set({ runId, byNode: {} });
  },

  handleEvent: (event: RunEventPayload) => {
    if (event.kind === "RunStarted") {
      get().beginRun(event.runId);
      return;
    }
    if (TERMINAL_KINDS.has(event.kind)) {
      // A terminal run never delivers more artifacts — pending previews
      // are stale, not errors.
      const byNode = get().byNode;
      const next: Record<string, ArtifactPreview> = {};
      for (const [nodeId, preview] of Object.entries(byNode)) {
        next[nodeId] =
          preview.status === "pending"
            ? { ...preview, status: "stale" }
            : preview;
      }
      set({ byNode: next });
      return;
    }
    if (!ARTIFACT_KINDS.has(event.kind)) return;

    const { nodeId, artifactId } = event;
    if (!nodeId || !artifactId) return;
    const runId = event.runId;
    const current = get();
    // Ignore artifacts from a previous run that resolved late.
    if (current.runId !== null && current.runId !== runId) return;

    const preview: ArtifactPreview = {
      artifactId,
      runId,
      nodeId,
      status: "pending",
    };
    set({ byNode: { ...current.byNode, [nodeId]: preview } });

    void resolveArtifact(artifactId)
      .then(async (metadata) => {
        // The run may have been superseded while resolving.
        if (get().runId !== null && get().runId !== runId) return;
        set({
          byNode: {
            ...get().byNode,
            [nodeId]: {
              ...preview,
              status: "ready",
              url: await artifactDisplayUrl(metadata),
            },
          },
        });
      })
      .catch((err: unknown) => {
        set({
          byNode: {
            ...get().byNode,
            [nodeId]: {
              ...preview,
              status: "error",
              error: String(err),
            },
          },
        });
      });
  },

  reset: () => set({ runId: null, byNode: {} }),
}));
