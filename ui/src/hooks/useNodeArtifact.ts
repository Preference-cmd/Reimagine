import { useArtifactStore, type ArtifactPreview } from "@/store/artifacts";

/**
 * Subscribe to the latest artifact preview for a node (F5-4).
 *
 * Returns undefined when the node produced no artifact in the current run.
 */
export function useNodeArtifact(nodeId: string): ArtifactPreview | undefined {
  return useArtifactStore((s) => s.byNode[nodeId]);
}
