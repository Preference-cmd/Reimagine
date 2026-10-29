import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { opInvoke, subscribeEvents } from "@/ipc";
import {
  DynamicWorkflowRunListSchema,
  type DynamicWorkflowRun,
  type SlotRunState,
} from "@/ipc/schemas";

/** Coalesces a burst of run notifications into a single read. */
const EVENT_DEBOUNCE_MS = 50;

/**
 * Latest dynamic workflow run for a collection and its per-slot states.
 * Events are notifications only, so each one triggers a fresh read.
 *
 * An event names its run (`documentId`) but not the collection that run
 * targets, and the collection may not have a run yet: an event for any run may
 * announce this collection's first or newest run. So every relevant event
 * schedules the same debounced read, and responses are applied only when they
 * still belong to the newest request: an older read that resolves late must
 * not roll the state back.
 */
export function useCollectionRun(
  projectId: string,
  collectionId: string | null | undefined,
): {
  run: DynamicWorkflowRun | null;
  slotStates: Record<string, SlotRunState>;
} {
  const [run, setRun] = useState<DynamicWorkflowRun | null>(null);
  const requestSeq = useRef(0);
  const debounceTimer = useRef<number | null>(null);

  const fetchRun = useCallback(async () => {
    const seq = ++requestSeq.current;
    if (!collectionId) {
      setRun(null);
      return;
    }
    try {
      const res = await opInvoke(projectId, "dynamic_workflow.list_runs", {
        collection_id: collectionId,
        limit: 1,
      });
      if (seq !== requestSeq.current) return;
      const parsed = DynamicWorkflowRunListSchema.parse(res.output);
      setRun(parsed.runs[0] ?? null);
    } catch {
      if (seq !== requestSeq.current) return;
      // A missing run list must not hide the collection itself.
      setRun(null);
    }
  }, [projectId, collectionId]);

  useEffect(() => {
    void fetchRun();
  }, [fetchRun]);

  useEffect(() => {
    const clearDebounce = () => {
      if (debounceTimer.current !== null) {
        window.clearTimeout(debounceTimer.current);
        debounceTimer.current = null;
      }
    };
    if (!collectionId) {
      clearDebounce();
      return;
    }
    const unsubscribe = subscribeEvents((event) => {
      if (event.type !== "document" || event.projectId !== projectId) return;
      if (
        event.kind !== "dynamic_workflow.slot_changed" &&
        event.kind !== "dynamic_workflow.run_changed"
      ) {
        return;
      }
      // The event may name the shown run or one this collection has not seen
      // yet; one fresh read of the latest run settles either case.
      if (debounceTimer.current !== null) return;
      debounceTimer.current = window.setTimeout(() => {
        debounceTimer.current = null;
        void fetchRun();
      }, EVENT_DEBOUNCE_MS);
    });
    return () => {
      unsubscribe();
      clearDebounce();
    };
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
