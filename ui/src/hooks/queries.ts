import { useQuery, useMutation } from "@tanstack/react-query";
import { listModels, searchModels, getModelCard, resolveArtifact } from "@/ipc";
import type { DownloadHuggingfaceModelArgs, ModelDownloadOutput } from "@/ipc/schemas";
import { downloadHuggingfaceModel as ipcDownloadModel } from "@/ipc";

/* ─── Query keys ─────────────────────────────────────────── */

export const queryKeys = {
  models: ["models"] as const,
  modelSearch: (query: string) => ["modelSearch", query] as const,
  modelCard: (repoId: string) => ["modelCard", repoId] as const,
  artifact: (artifactId: string) => ["artifact", artifactId] as const,
} as const;

/* ─── Queries ────────────────────────────────────────────── */

/** Installed models list — shared by the models page and the install dialog. */
export function useModels() {
  return useQuery({
    queryKey: queryKeys.models,
    queryFn: listModels,
    staleTime: 60_000,
  });
}

/** HuggingFace model search — enabled only when query is non-empty. */
export function useModelSearch(query: string) {
  return useQuery({
    queryKey: queryKeys.modelSearch(query),
    queryFn: () => searchModels(query),
    enabled: query.trim().length > 0,
    staleTime: 30_000,
  });
}

/** Model card detail — enabled only when repoId is provided. */
export function useModelCard(repoId: string | null) {
  return useQuery({
    queryKey: queryKeys.modelCard(repoId ?? ""),
    queryFn: () => getModelCard(repoId!),
    enabled: !!repoId,
    staleTime: Infinity,
  });
}

/** Artifact metadata resolution — enabled only when artifactId is provided. */
export function useArtifactQuery(artifactId: string | null) {
  return useQuery({
    queryKey: queryKeys.artifact(artifactId ?? ""),
    queryFn: () => resolveArtifact(artifactId!),
    enabled: !!artifactId,
    staleTime: Infinity,
  });
}

/* ─── Mutations ──────────────────────────────────────────── */

/** Download a HuggingFace model; progress arrives as `aigc.download.*` events. */
export function useDownloadModel() {
  return useMutation({
    mutationFn: (args: DownloadHuggingfaceModelArgs): Promise<ModelDownloadOutput> =>
      ipcDownloadModel(args),
  });
}
