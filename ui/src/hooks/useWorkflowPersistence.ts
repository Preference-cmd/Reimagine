import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useWorkflowStore } from "@/store/workflow";
import { useRecentWorkflowsStore } from "@/store/recentWorkflows";
import { saveWorkflow, loadWorkflow } from "@/ipc";
import { workflowFromJson, workflowToJson } from "@/lib/workflowCodec";
import { queryKeys } from "@/hooks/queries";

/**
 * Workflow persistence (F1-2):
 *   - Cmd/Ctrl+S saves immediately
 *   - changes to nodes/edges auto-save after a 5s debounce
 *   - on mount, loads the most recent saved workflow (if any)
 *
 * Uses TanStack Query cache for the workflow list so subsequent
 * components can read it without re-fetching.
 */
const AUTOSAVE_DEBOUNCE_MS = 5000;

let autosaveTimer: ReturnType<typeof setTimeout> | null = null;
let saveInFlight = false;

/** Save the current workflow now (shared by the hook, Cmd+S, and the TopBar button). */
export async function saveWorkflowNow(queryClient?: {
  invalidateQueries: (args: { queryKey: readonly unknown[] }) => Promise<void>;
}): Promise<void> {
  if (autosaveTimer !== null) {
    clearTimeout(autosaveTimer);
    autosaveTimer = null;
  }
  if (saveInFlight) return;
  const { nodes, edges, id, name } = useWorkflowStore.getState();
  saveInFlight = true;
  try {
    await saveWorkflow(id, workflowToJson(nodes, edges, id, name));
    useRecentWorkflowsStore.getState().addRecent(id, name);
    // Invalidate the workflows list cache so other consumers see fresh data
    await queryClient?.invalidateQueries({ queryKey: queryKeys.workflows });
  } catch (error) {
    console.error("[persistence] save failed:", error);
  } finally {
    saveInFlight = false;
  }
}

export function useWorkflowPersistence() {
  const queryClient = useQueryClient();

  // Load the most recent saved workflow on app start. If none exists (or the
  // load fails), the initial demo graph remains as the default content.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        // Use the query cache — populates it for other consumers
        const summaries = await queryClient.fetchQuery({
          queryKey: queryKeys.workflows,
          queryFn: () => import("@/ipc").then((m) => m.listWorkflows()),
          staleTime: Infinity,
        });
        if (cancelled) return;
        const mostRecent = summaries[0];
        if (!mostRecent) return;
        const json = await loadWorkflow(mostRecent.id);
        if (cancelled) return;
        const { nodes, edges, name } = workflowFromJson(json);
        useWorkflowStore.getState().hydrate(nodes, edges, mostRecent.id, name);
        useRecentWorkflowsStore.getState().addRecent(mostRecent.id, name);
        // The pre-load demo graph must not be reachable via undo.
        useWorkflowStore.temporal.getState().clear();
      } catch (error) {
        console.warn("[persistence] initial load failed; using demo workflow", error);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [queryClient]);

  // Cmd/Ctrl+S — immediate save.
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const mod = event.metaKey || event.ctrlKey;
      if (!mod || event.key.toLowerCase() !== "s") return;
      event.preventDefault();
      void saveWorkflowNow(queryClient);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [queryClient]);

  // Auto-save — debounced 5s after the last nodes/edges change.
  useEffect(() => {
    const unsubscribe = useWorkflowStore.subscribe((state, prev) => {
      if (state.nodes === prev.nodes && state.edges === prev.edges) return;
      if (autosaveTimer !== null) clearTimeout(autosaveTimer);
      autosaveTimer = setTimeout(() => {
        autosaveTimer = null;
        void saveWorkflowNow(queryClient);
      }, AUTOSAVE_DEBOUNCE_MS);
    });
    return () => {
      unsubscribe();
      if (autosaveTimer !== null) {
        clearTimeout(autosaveTimer);
        autosaveTimer = null;
      }
    };
  }, [queryClient]);
}
