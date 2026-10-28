import { useCallback, useEffect, useMemo, useState } from "react";
import { opInvoke, subscribeEvents } from "@/ipc";
import {
  DynamicWorkflowRunListSchema,
  type DynamicWorkflowRun,
  type SlotRunState,
} from "@/ipc/schemas";

/**
 * Latest dynamic workflow run for a collection and its per-slot states.
 * Events are notifications only, so each one triggers a fresh read.
 */
export function useCollectionRun(
  projectId: string,
  collectionId: string | null | undefined,
): {
  run: DynamicWorkflowRun | null;
  slotStates: Record<string, SlotRunState>;
} {
  const [run, setRun] = useState<DynamicWorkflowRun | null>(null);

  const fetchRun = useCallback(async () => {
    if (!collectionId) {
      setRun(null);
      return;
    }
    try {
      const res = await opInvoke(projectId, "dynamic_workflow.list_runs", {
        collection_id: collectionId,
      });
      const parsed = DynamicWorkflowRunListSchema.parse(res.output);
      setRun(parsed.runs[0] ?? null);
    } catch {
      // A missing run list must not hide the collection itself.
      setRun(null);
    }
  }, [projectId, collectionId]);

  useEffect(() => {
    void fetchRun();
  }, [fetchRun]);

  useEffect(() => {
    if (!collectionId) return;
    return subscribeEvents((event) => {
      if (event.type !== "document" || event.projectId !== projectId) return;
      if (
        event.kind === "dynamic_workflow.slot_changed" ||
        event.kind === "dynamic_workflow.run_changed"
      ) {
        void fetchRun();
      }
    });
  }, [projectId, collectionId, fetchRun]);

  const slotStates = useMemo(() => {
    const states: Record<string, SlotRunState> = {};
    for (const item of run?.items ?? []) {
      states[item.slot_id ?? item.item_id] = item.state;
    }
    return states;
  }, [run]);

  return { run, slotStates };
}
