import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getNodeDefs,
  listModels,
  searchModels,
  getModelCard,
  resolveArtifact,
  listWorkflows,
  saveWorkflow,
  listAgentProviders,
} from "@/ipc";
import type {
  DownloadHuggingfaceModelArgs,
  ModelDownloadOutput,
  DownloadEventPayload,
} from "@/ipc/schemas";
import { downloadHuggingfaceModel as ipcDownloadModel } from "@/ipc";

/* ─── Query keys ─────────────────────────────────────────── */

export const queryKeys = {
  nodeDefs: ["nodeDefs"] as const,
  models: ["models"] as const,
  modelSearch: (query: string) => ["modelSearch", query] as const,
  modelCard: (repoId: string) => ["modelCard", repoId] as const,
  artifact: (artifactId: string) => ["artifact", artifactId] as const,
  workflows: ["workflows"] as const,
  agentProviders: ["agentProviders"] as const,
} as const;

/* ─── Queries ────────────────────────────────────────────── */

/** Node catalog — loaded once at startup, never revalidated. */
export function useNodeDefs() {
  return useQuery({
    queryKey: queryKeys.nodeDefs,
    queryFn: getNodeDefs,
    staleTime: Infinity,
  });
}

/** Installed models list — shared between ModelsView and ExplorerPanel. */
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

/** Saved workflows list — one-shot load. */
export function useWorkflows() {
  return useQuery({
    queryKey: queryKeys.workflows,
    queryFn: listWorkflows,
    staleTime: Infinity,
  });
}

/** Agent providers — prepare for future agent UI. */
export function useAgentProviders() {
  return useQuery({
    queryKey: queryKeys.agentProviders,
    queryFn: listAgentProviders,
    staleTime: Infinity,
  });
}

/* ─── Mutations ──────────────────────────────────────────── */

/** Save workflow — invalidates the workflows list on success. */
export function useSaveWorkflow() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, json }: { id: string; json: unknown }) => saveWorkflow(id, json),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.workflows });
    },
  });
}

/** Download a HuggingFace model with streaming progress. */
export function useDownloadModel() {
  return useMutation({
    mutationFn: ({
      args,
      onEvent,
    }: {
      args: DownloadHuggingfaceModelArgs;
      onEvent?: (event: DownloadEventPayload) => void;
    }): Promise<ModelDownloadOutput> => ipcDownloadModel(args, onEvent),
  });
}
