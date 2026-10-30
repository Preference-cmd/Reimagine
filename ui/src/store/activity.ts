import { create } from "zustand";
import type { ActivityRecord } from "@/ipc/schemas";
import {
  ACTIVITY_PAGE_SIZE,
  applyActivityRecord,
  canRedo,
  canUndo,
  listActivity,
  mergeActivityPage,
  redoEntryId,
  undoEntryId,
  type ActivityStacks,
} from "@/lib/activity";

export type ActivityState = ActivityStacks & {
  /** Project whose feed is loaded; null before the first load. */
  projectId: string | null;
  /** Newest first, as the host pages them. */
  records: ActivityRecord[];
  hasMore: boolean;
  loading: boolean;
  loadingMore: boolean;
  /** True once the first page resolved; keeps remounts from refetching. */
  loaded: boolean;
  error: string | null;
  load: (projectId: string) => Promise<void>;
  loadMore: () => Promise<void>;
  append: (record: ActivityRecord) => void;
  reset: () => void;
};

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function emptyState() {
  return {
    projectId: null,
    records: [] as ActivityRecord[],
    hasMore: false,
    loading: false,
    loadingMore: false,
    error: null,
    undoStack: [] as string[],
    redoStack: [] as string[],
    loaded: false,
  };
}

/** Invalidates in-flight reads after a project switch or reset. */
let readSequence = 0;

/**
 * Workspace activity feed for the active project. The shell mounts one sync
 * hook; the panel reads this store when it opens.
 *
 * Loaded pages only fill `records`. The undo/redo stacks start empty and
 * change solely through `append`: the board history lives in memory, so
 * history from before a restart has nothing to target.
 */
export const useActivityStore = create<ActivityState>()((set, get) => ({
  ...emptyState(),

  /** First page for `projectId`; a reload of the same project is a no-op. */
  load: async (projectId) => {
    const current = get();
    if (current.projectId === projectId && (current.loading || current.loaded)) return;

    const sequence = ++readSequence;
    set({ ...emptyState(), projectId, loading: true });

    try {
      const page = await listActivity(projectId, { limit: ACTIVITY_PAGE_SIZE });
      if (sequence !== readSequence || get().projectId !== projectId) return;
      set((state) => ({
        records: mergeActivityPage(state.records, page.records),
        hasMore: page.hasMore,
        loading: false,
        loaded: true,
      }));
    } catch (error) {
      if (sequence !== readSequence || get().projectId !== projectId) return;
      set({ loading: false, error: errorMessage(error) });
    }
  },

  /** Older page after the last loaded record; appends and dedupes by id. */
  loadMore: async () => {
    const state = get();
    if (!state.projectId || !state.hasMore || state.loading || state.loadingMore) return;
    const before = state.records[state.records.length - 1]?.id;
    if (!before) return;

    const projectId = state.projectId;
    const sequence = ++readSequence;
    set({ loadingMore: true, error: null });

    try {
      const page = await listActivity(projectId, {
        limit: ACTIVITY_PAGE_SIZE,
        before,
      });
      if (sequence !== readSequence || get().projectId !== projectId) return;
      set((current) => ({
        records: mergeActivityPage(current.records, page.records),
        hasMore: page.hasMore,
        loadingMore: false,
      }));
    } catch (error) {
      if (sequence !== readSequence || get().projectId !== projectId) return;
      set({ loadingMore: false, error: errorMessage(error) });
    }
  },

  /** Prepend a live record; duplicate ids are ignored. */
  append: (record) => {
    const state = get();
    if (state.records.some((existing) => existing.id === record.id)) return;

    const stacks = applyActivityRecord(state, record);
    set({
      records: [record, ...state.records],
      undoStack: stacks.undoStack,
      redoStack: stacks.redoStack,
    });
  },

  reset: () => {
    readSequence += 1;
    set(emptyState());
  },
}));

/* ───── Selectors for the activity panel ───── */

export const selectUndoEntryId = (state: ActivityState) => undoEntryId(state);
export const selectRedoEntryId = (state: ActivityState) => redoEntryId(state);
export const selectCanUndo = (state: ActivityState) => canUndo(state);
export const selectCanRedo = (state: ActivityState) => canRedo(state);
