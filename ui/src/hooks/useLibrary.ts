import { useCallback, useEffect, useMemo } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import * as m from "$paraglide/messages";
import { subscribeEvents } from "@/ipc";
import type { RunState } from "@/ipc/schemas";
import {
  buildLibraryRows,
  countLibraryRows,
  listAssets,
  listCollections,
  listDynamicWorkflows,
  listRuns,
  type LibraryFilter,
  type LibraryRow,
  type LibraryRowLabels,
} from "@/lib/library";

/** Query keys per project, so a workspace switch never serves another's rows. */
export const libraryQueryKeys = {
  project: (projectId: string) => ["library", projectId] as const,
  collections: (projectId: string) => ["library", projectId, "collections"] as const,
  workflows: (projectId: string) => ["library", projectId, "workflows"] as const,
  runs: (projectId: string, limit: number) => ["library", projectId, "runs", limit] as const,
  assets: (projectId: string) => ["library", projectId, "assets"] as const,
};

const RUN_STATE_LABELS: Record<RunState, () => string> = {
  queued: m["library.runState.queued"],
  running: m["library.runState.running"],
  completed: m["library.runState.completed"],
  failed: m["library.runState.failed"],
  cancelled: m["library.runState.cancelled"],
  interrupted: m["library.runState.interrupted"],
  needs_confirmation: m["library.runState.needsConfirmation"],
};

const LABELS: LibraryRowLabels = {
  collectionDetail: (count, version) => `${count} ${m["library.slots"]()} · v${version}`,
  workflowDetail: (count, version) => `${count} ${m["library.steps"]()} · v${version}`,
  runState: (state) => RUN_STATE_LABELS[state](),
};

/**
 * The library's four list reads plus the combined table rows. Lists refetch
 * on every mount and follow document events for the active project:
 * `collection.changed` refreshes collections and `dynamic_workflow.*`
 * refreshes workflows and runs.
 */
export function useLibraryRows(
  projectId: string,
  limit = 50,
): {
  rows: LibraryRow[];
  counts: Record<LibraryFilter, number>;
  isLoading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
} {
  const queryClient = useQueryClient();

  const collections = useQuery({
    queryKey: libraryQueryKeys.collections(projectId),
    queryFn: () => listCollections(projectId),
    staleTime: 0,
    refetchOnMount: "always",
  });
  const workflows = useQuery({
    queryKey: libraryQueryKeys.workflows(projectId),
    queryFn: () => listDynamicWorkflows(projectId),
    staleTime: 0,
    refetchOnMount: "always",
  });
  const runs = useQuery({
    queryKey: libraryQueryKeys.runs(projectId, limit),
    queryFn: () => listRuns(projectId, limit),
    staleTime: 0,
    refetchOnMount: "always",
  });
  const assets = useQuery({
    queryKey: libraryQueryKeys.assets(projectId),
    queryFn: () => listAssets(projectId),
    staleTime: 0,
    refetchOnMount: "always",
  });

  useEffect(() => {
    if (!projectId) return;
    const unsubscribe = subscribeEvents((event) => {
      if (event.type !== "document" || event.projectId !== projectId) return;
      if (event.kind === "collection.changed") {
        void queryClient.invalidateQueries({
          queryKey: libraryQueryKeys.collections(projectId),
        });
      } else if (event.kind.startsWith("dynamic_workflow.")) {
        void queryClient.invalidateQueries({
          queryKey: libraryQueryKeys.workflows(projectId),
        });
        void queryClient.invalidateQueries({
          queryKey: libraryQueryKeys.runs(projectId, limit),
        });
      }
    });
    return () => unsubscribe();
  }, [projectId, limit, queryClient]);

  const rows = useMemo(
    () =>
      buildLibraryRows(
        {
          collections: collections.data ?? [],
          workflows: workflows.data ?? [],
          runs: runs.data ?? [],
          assets: assets.data ?? [],
        },
        LABELS,
      ),
    [collections.data, workflows.data, runs.data, assets.data],
  );

  const counts = useMemo(() => countLibraryRows(rows), [rows]);
  const queries = [collections, workflows, runs, assets];
  const isLoading = queries.some((query) => query.isLoading);
  const error = queries.find((query) => query.error)?.error ?? null;

  const refetch = useCallback(async () => {
    await queryClient.invalidateQueries({ queryKey: libraryQueryKeys.project(projectId) });
  }, [queryClient, projectId]);

  return { rows, counts, isLoading, error, refetch };
}
