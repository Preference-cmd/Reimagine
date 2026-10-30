import { create } from "zustand";
import type { ActivityRecord } from "@/ipc/schemas";
import {
  ACTIVITY_PAGE_SIZE,
  applyActivityRecord,
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
  /**
   * First page for `projectId`; a reload of the same project is a no-op
   * unless `force` is set. A forced load refetches the page while keeping the
   * project's stacks.
   */
  load: (projectId: string, options?: { force?: boolean }) => Promise<void>;
  loadMore: () => Promise<void>;
  append: (projectId: string, record: ActivityRecord) => void;
  /**
   * Replace a project's stacks with the host's current tops after a rejected
   * undo/redo. The host is authoritative here: its entry is the only one the
   * next request can act on.
   */
  resyncHistory: (projectId: string, tops: ActivityHistoryResync) => void;
  reset: () => void;
};

/** Host-side tops after the feed and the board history disagreed. */
export type ActivityHistoryResync = {
  undoEntryId: string | null;
  redoEntryId: string | null;
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
 * Undo/redo stacks per project for this app session. The board history is
 * per-project and survives a workspace switch, so the stacks the feed knows
 * about must survive it too; `load` restores the entry for the project it
 * opens.
 */
const projectStacks = new Map<string, ActivityStacks>();

function emptyStacks(): ActivityStacks {
  return { undoStack: [], redoStack: [] };
}

function stacksFor(projectId: string | null): ActivityStacks {
  if (!projectId) return emptyStacks();
  return projectStacks.get(projectId) ?? emptyStacks();
}

function rememberStacks(projectId: string, stacks: ActivityStacks): void {
  if (stacks.undoStack.length === 0 && stacks.redoStack.length === 0) {
    projectStacks.delete(projectId);
  } else {
    projectStacks.set(projectId, stacks);
  }
}

/**
 * Workspace activity feed for the active project. The shell mounts one sync
 * hook; the panel reads this store when it opens.
 *
 * Loaded pages only fill `records`. The undo/redo stacks start empty and
 * change solely through `append`, or through an explicit `resyncHistory` when
 * the host rejects a stale top: the board history lives in memory, so history
 * from before a restart has nothing to target. Loaded pages never seed the
 * stacks by themselves, but the stacks a project accrued during the session
 * are restored when the feed returns to it.
 */
export const useActivityStore = create<ActivityState>()((set, get) => ({
  ...emptyState(),

  /** First page for `projectId`; reloads only when forced. */
  load: async (projectId, options = {}) => {
    const current = get();
    if (!options.force && current.projectId === projectId && (current.loading || current.loaded)) {
      return;
    }

    const sequence = ++readSequence;
    const stacks = stacksFor(projectId);
    set({
      ...emptyState(),
      projectId,
      loading: true,
      undoStack: stacks.undoStack,
      redoStack: stacks.redoStack,
    });

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

  /**
   * Prepend a live record for `projectId`; duplicate ids are ignored. A
   * record for a project other than the loaded one only updates that
   * project's stacks, so a switch cannot apply the wrong stacks or mix the
   * feeds. With no project loaded the feed stays empty and only the stacks
   * are remembered; the next `load` restores them.
   */
  append: (projectId, record) => {
    const state = get();
    const active = state.projectId === projectId;
    if (active && state.records.some((existing) => existing.id === record.id)) {
      return;
    }

    const stacks = applyActivityRecord(active ? state : stacksFor(projectId), record);
    rememberStacks(projectId, stacks);
    if (!active) return;

    set({
      projectId,
      records: [record, ...state.records],
      undoStack: stacks.undoStack,
      redoStack: stacks.redoStack,
    });
  },

  /** Replace a project's stacks with the host's current tops. */
  resyncHistory: (projectId, tops) => {
    const stacks: ActivityStacks = {
      undoStack: tops.undoEntryId ? [tops.undoEntryId] : [],
      redoStack: tops.redoEntryId ? [tops.redoEntryId] : [],
    };
    rememberStacks(projectId, stacks);
    if (get().projectId !== projectId) return;
    set({ undoStack: stacks.undoStack, redoStack: stacks.redoStack });
  },

  /** Clears the feed and the session's per-project stacks. */
  reset: () => {
    readSequence += 1;
    projectStacks.clear();
    set(emptyState());
  },
}));

/* ───── Selectors for the activity panel ───── */

export const selectUndoEntryId = (state: ActivityState) => undoEntryId(state);
export const selectRedoEntryId = (state: ActivityState) => redoEntryId(state);
