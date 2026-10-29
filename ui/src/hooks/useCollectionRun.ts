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
 * Events name their run (`documentId`), so only notifications about the run
 * currently shown cause a read, and responses are applied only when they still
 * belong to the newest request: an older read that resolves late must not
 * roll the state back.
 */
export function useCollectionRun(
  projectId: string,
  collectionId: string | null | undefined,
): {
  run: DynamicWorkflowRun | null;
  slotStates: Record<string, SlotRunState>;
} {
  const [run, setRun] = useState<DynamicWorkflowRun | null>(null);
  const currentRunId = useRef<string | null>(null);
  const requestSeq = useRef(0);
  const debounceTimer = useRef<number | null>(null);

  const fetchRun = useCallback(async () => {
    const seq = ++requestSeq.current;
    if (!collectionId) {
      currentRunId.current = null;
      setRun(null);
      return;
    }
    try {
      const res = await opInvoke(projectId, "dynamic_workflow.list_runs", {
        collection_id: collectionId,
      });
      if (seq !== requestSeq.current) return;
      const parsed = DynamicWorkflowRunListSchema.parse(res.output);
      currentRunId.current = parsed.runs[0]?.id ?? null;
      setRun(parsed.runs[0] ?? null);
    } catch {
      if (seq !== requestSeq.current) return;
      // A missing run list must not hide the collection itself.
      currentRunId.current = null;
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
      // A notification about another run says nothing about the run shown.
      if (event.documentId !== currentRunId.current) return;
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
