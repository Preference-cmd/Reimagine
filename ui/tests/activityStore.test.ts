import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { act, renderHook, waitFor } from "./test-utils";
import * as ipc from "../src/ipc";
import { ActivityListResultSchema, type ActivityRecord } from "../src/ipc/schemas";
import {
  ACTIVITY_PAGE_SIZE,
  activityActorMatches,
  applyActivityRecord,
  canRedo,
  canUndo,
  isActivityActorFilter,
  mergeActivityPage,
  redoEntryId,
  undoEntryId,
  type ActivityStacks,
} from "../src/lib/activity";
import { useActivityStore } from "../src/store/activity";
import { useActivitySync } from "../src/hooks/useActivitySync";

let sequence = 0;

function record(overrides: Partial<ActivityRecord> = {}): ActivityRecord {
  sequence += 1;
  return {
    id: `act-${sequence}`,
    at: "2026-10-30T00:00:00Z",
    actor: { kind: "Human", id: "human-1", label: "You" },
    summary: "Added a note",
    summaryKey: "activity.board.addNote",
    summaryParams: {},
    targets: ["note-1"],
    kind: "applied",
    historyEntryId: null,
    ...overrides,
  };
}

function pageOutput(records: ActivityRecord[], hasMore: boolean) {
  return { step: {}, output: { records, has_more: hasMore } };
}

function ids(records: ActivityRecord[]): string[] {
  return records.map((entry) => entry.id);
}

const EMPTY_STACKS: ActivityStacks = { undoStack: [], redoStack: [] };

beforeEach(() => {
  useActivityStore.getState().reset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("activity stack transitions", () => {
  test("applied pushes onto undo, clears redo and leaves the input untouched", () => {
    const current = { undoStack: ["h-1"], redoStack: ["h-2"] };
    const next = applyActivityRecord(current, record({ kind: "applied", historyEntryId: "h-3" }));

    expect(next).toEqual({ undoStack: ["h-1", "h-3"], redoStack: [] });
    expect(current).toEqual({ undoStack: ["h-1"], redoStack: ["h-2"] });
  });

  test("undone removes the entry from undo and pushes it onto redo", () => {
    const next = applyActivityRecord(
      { undoStack: ["h-1", "h-2"], redoStack: [] },
      record({ kind: "undone", historyEntryId: "h-2" }),
    );
    expect(next).toEqual({ undoStack: ["h-1"], redoStack: ["h-2"] });
  });

  test("redone removes the entry from redo and pushes it back onto undo", () => {
    const next = applyActivityRecord(
      { undoStack: ["h-1"], redoStack: ["h-3", "h-2"] },
      record({ kind: "redone", historyEntryId: "h-2" }),
    );
    expect(next).toEqual({ undoStack: ["h-1", "h-2"], redoStack: ["h-3"] });
  });

  test("multi-step apply/undo/redo keeps both tops in order", () => {
    let stacks = EMPTY_STACKS;
    stacks = applyActivityRecord(stacks, record({ kind: "applied", historyEntryId: "h-1" }));
    stacks = applyActivityRecord(stacks, record({ kind: "applied", historyEntryId: "h-2" }));
    expect(undoEntryId(stacks)).toBe("h-2");

    stacks = applyActivityRecord(stacks, record({ kind: "undone", historyEntryId: "h-2" }));
    expect(stacks).toEqual({ undoStack: ["h-1"], redoStack: ["h-2"] });
    expect(redoEntryId(stacks)).toBe("h-2");

    stacks = applyActivityRecord(stacks, record({ kind: "undone", historyEntryId: "h-1" }));
    expect(stacks).toEqual({ undoStack: [], redoStack: ["h-2", "h-1"] });

    stacks = applyActivityRecord(stacks, record({ kind: "redone", historyEntryId: "h-1" }));
    expect(stacks).toEqual({ undoStack: ["h-1"], redoStack: ["h-2"] });

    stacks = applyActivityRecord(stacks, record({ kind: "redone", historyEntryId: "h-2" }));
    expect(stacks).toEqual({ undoStack: ["h-1", "h-2"], redoStack: [] });
    expect(canUndo(stacks)).toBe(true);
    expect(canRedo(stacks)).toBe(false);
  });

  test("records without a history entry id leave the stacks untouched", () => {
    const current = { undoStack: ["h-1"], redoStack: ["h-2"] };
    for (const kind of ["applied", "undone", "redone"] as const) {
      expect(applyActivityRecord(current, record({ kind, historyEntryId: null }))).toBe(current);
    }
  });

  test("run records never touch the stacks", () => {
    const current = { undoStack: ["h-1"], redoStack: [] };
    for (const kind of ["run_started", "run_finished", "run_cancelled"] as const) {
      expect(applyActivityRecord(current, record({ kind, historyEntryId: "h-9" }))).toBe(current);
    }
  });

  test("stack helpers report empty tops", () => {
    expect(undoEntryId(EMPTY_STACKS)).toBeNull();
    expect(redoEntryId(EMPTY_STACKS)).toBeNull();
    expect(canUndo(EMPTY_STACKS)).toBe(false);
    expect(canRedo(EMPTY_STACKS)).toBe(false);
  });
});

describe("activity page merge and actor filter", () => {
  test("mergeActivityPage appends older records and drops duplicates", () => {
    const a = record({ id: "a" });
    const b = record({ id: "b" });
    const c = record({ id: "c" });

    expect(ids(mergeActivityPage([a, b], [b, c]))).toEqual(["a", "b", "c"]);
  });

  test("mergeActivityPage keeps the current array when nothing is new", () => {
    const current = [record({ id: "a" }), record({ id: "b" })];
    expect(mergeActivityPage(current, [])).toBe(current);
    expect(mergeActivityPage(current, [current[0]])).toBe(current);
  });

  test("activity.list pages accept the host's has_more and expose hasMore", () => {
    const entry = record({ id: "act-1" });
    expect(ActivityListResultSchema.parse({ records: [entry], has_more: true })).toEqual({
      records: [entry],
      hasMore: true,
    });
    expect(ActivityListResultSchema.parse({ records: [], hasMore: false })).toEqual({
      records: [],
      hasMore: false,
    });
  });

  test("activityActorMatches separates you, agent and everyone", () => {
    const human = { actor: { kind: "Human" as const, id: null, label: null } };
    const agent = { actor: { kind: "Agent" as const, id: "agent-1", label: "Agent" } };
    const importer = { actor: { kind: "Importer" as const, id: null, label: null } };

    expect(activityActorMatches(human, "all")).toBe(true);
    expect(activityActorMatches(human, "you")).toBe(true);
    expect(activityActorMatches(human, "agent")).toBe(false);
    expect(activityActorMatches(agent, "all")).toBe(true);
    expect(activityActorMatches(agent, "you")).toBe(false);
    expect(activityActorMatches(agent, "agent")).toBe(true);
    expect(activityActorMatches(importer, "you")).toBe(false);
    expect(activityActorMatches(importer, "agent")).toBe(false);
  });

  test("isActivityActorFilter rejects unknown filters", () => {
    expect(isActivityActorFilter("you")).toBe(true);
    expect(isActivityActorFilter("agent")).toBe(true);
    expect(isActivityActorFilter("system")).toBe(false);
    expect(isActivityActorFilter(null)).toBe(false);
  });
});

describe("activity store", () => {
  test("load fetches the first page but never creates undo targets", async () => {
    const newer = record({ id: "act-3", historyEntryId: "h-3" });
    const older = record({ id: "act-2", historyEntryId: "h-2" });
    const opInvoke = vi
      .spyOn(ipc, "opInvoke")
      .mockResolvedValueOnce(pageOutput([newer, older], true));

    await useActivityStore.getState().load("proj-1");

    expect(opInvoke).toHaveBeenCalledWith("proj-1", "activity.list", {
      limit: ACTIVITY_PAGE_SIZE,
    });
    const state = useActivityStore.getState();
    expect(state.projectId).toBe("proj-1");
    expect(ids(state.records)).toEqual(["act-3", "act-2"]);
    expect(state.hasMore).toBe(true);
    expect(state.loading).toBe(false);
    expect(state.loaded).toBe(true);
    expect(state.error).toBeNull();
    expect(state.undoStack).toEqual([]);
    expect(state.redoStack).toEqual([]);
  });

  test("load is a no-op for the project already loaded", async () => {
    const opInvoke = vi.spyOn(ipc, "opInvoke").mockResolvedValue(pageOutput([record()], false));

    await useActivityStore.getState().load("proj-1");
    await useActivityStore.getState().load("proj-1");

    expect(opInvoke).toHaveBeenCalledTimes(1);
  });

  test("a failed first load is recorded and a later load retries", async () => {
    const opInvoke = vi
      .spyOn(ipc, "opInvoke")
      .mockRejectedValueOnce(new Error("host down"))
      .mockResolvedValueOnce(pageOutput([record({ id: "act-9" })], false));

    await useActivityStore.getState().load("proj-1");
    expect(useActivityStore.getState().error).toBe("host down");
    expect(useActivityStore.getState().loaded).toBe(false);

    await useActivityStore.getState().load("proj-1");
    expect(opInvoke).toHaveBeenCalledTimes(2);
    expect(useActivityStore.getState().error).toBeNull();
    expect(ids(useActivityStore.getState().records)).toEqual(["act-9"]);
  });

  test("loadMore pages from the last id, appends older records and dedupes", async () => {
    const c = record({ id: "c" });
    const b = record({ id: "b" });
    const a = record({ id: "a" });
    const opInvoke = vi
      .spyOn(ipc, "opInvoke")
      .mockResolvedValueOnce(pageOutput([c, b], true))
      .mockResolvedValueOnce(pageOutput([b, a], false));

    await useActivityStore.getState().load("proj-1");
    await useActivityStore.getState().loadMore();

    expect(opInvoke).toHaveBeenLastCalledWith("proj-1", "activity.list", {
      limit: ACTIVITY_PAGE_SIZE,
      before: "b",
    });
    const state = useActivityStore.getState();
    expect(ids(state.records)).toEqual(["c", "b", "a"]);
    expect(state.hasMore).toBe(false);
    expect(state.loadingMore).toBe(false);

    // Nothing older remains: another call must not hit the host.
    await useActivityStore.getState().loadMore();
    expect(opInvoke).toHaveBeenCalledTimes(2);
  });

  test("loadMore does nothing before a project is loaded", async () => {
    const opInvoke = vi.spyOn(ipc, "opInvoke").mockResolvedValue(pageOutput([], false));

    await useActivityStore.getState().loadMore();

    expect(opInvoke).not.toHaveBeenCalled();
  });

  test("a project change clears records and stacks before loading", async () => {
    const first = record({ id: "act-1", historyEntryId: "h-1" });
    const second = record({ id: "act-2", historyEntryId: "h-2" });
    vi.spyOn(ipc, "opInvoke")
      .mockResolvedValueOnce(pageOutput([first], false))
      .mockResolvedValueOnce(pageOutput([second], false));

    await useActivityStore.getState().load("proj-1");
    useActivityStore
      .getState()
      .append(record({ id: "live-1", kind: "applied", historyEntryId: "h-live" }));
    expect(useActivityStore.getState().undoStack).toEqual(["h-live"]);

    await useActivityStore.getState().load("proj-2");

    const state = useActivityStore.getState();
    expect(state.projectId).toBe("proj-2");
    expect(ids(state.records)).toEqual(["act-2"]);
    expect(state.undoStack).toEqual([]);
    expect(state.redoStack).toEqual([]);
  });

  test("append prepends live records once and moves the stacks", () => {
    const applied = record({ id: "live-1", kind: "applied", historyEntryId: "h-1" });
    useActivityStore.getState().append(applied);
    useActivityStore.getState().append(applied);

    expect(ids(useActivityStore.getState().records)).toEqual(["live-1"]);
    expect(useActivityStore.getState().undoStack).toEqual(["h-1"]);

    useActivityStore
      .getState()
      .append(record({ id: "live-2", kind: "undone", historyEntryId: "h-1" }));
    expect(ids(useActivityStore.getState().records)).toEqual(["live-2", "live-1"]);
    expect(useActivityStore.getState().undoStack).toEqual([]);
    expect(useActivityStore.getState().redoStack).toEqual(["h-1"]);

    useActivityStore
      .getState()
      .append(record({ id: "run-1", kind: "run_started", historyEntryId: null }));
    expect(useActivityStore.getState().undoStack).toEqual([]);
    expect(useActivityStore.getState().redoStack).toEqual(["h-1"]);
  });

  test("reset clears the feed, error and stacks", async () => {
    vi.spyOn(ipc, "opInvoke").mockResolvedValueOnce(
      pageOutput([record({ id: "act-1", historyEntryId: "h-1" })], true),
    );
    await useActivityStore.getState().load("proj-1");
    useActivityStore
      .getState()
      .append(record({ id: "live-1", kind: "applied", historyEntryId: "h-2" }));

    useActivityStore.getState().reset();

    const state = useActivityStore.getState();
    expect(state.projectId).toBeNull();
    expect(state.records).toEqual([]);
    expect(state.hasMore).toBe(false);
    expect(state.loaded).toBe(false);
    expect(state.error).toBeNull();
    expect(state.undoStack).toEqual([]);
    expect(state.redoStack).toEqual([]);
  });
});

describe("useActivitySync", () => {
  test("loads the project and appends only its live activity records", async () => {
    const listeners = new Set<(event: ipc.Event) => void>();
    vi.spyOn(ipc, "subscribeEvents").mockImplementation((listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    });
    const opInvoke = vi
      .spyOn(ipc, "opInvoke")
      .mockResolvedValue(pageOutput([record({ id: "act-page" })], false));

    const { unmount } = renderHook(() => useActivitySync("proj-1"));
    await waitFor(() => expect(useActivityStore.getState().loaded).toBe(true));

    expect(opInvoke).toHaveBeenCalledWith("proj-1", "activity.list", {
      limit: ACTIVITY_PAGE_SIZE,
    });
    expect(listeners.size).toBe(1);

    const live = record({ id: "live-1", kind: "applied", historyEntryId: "h-live" });
    act(() => {
      for (const listener of listeners) {
        listener({
          type: "mod",
          kind: "activity.appended",
          payload: { projectId: "proj-1", record: live },
        });
      }
    });

    expect(useActivityStore.getState().records[0]?.id).toBe("live-1");
    expect(useActivityStore.getState().undoStack).toEqual(["h-live"]);

    act(() => {
      for (const listener of listeners) {
        listener({
          type: "mod",
          kind: "activity.appended",
          payload: { projectId: "proj-2", record: record({ id: "live-2" }) },
        });
        listener({ type: "mod", kind: "board.changed", payload: {} });
        listener({ type: "mod", kind: "activity.appended", payload: { projectId: "proj-1" } });
      }
    });

    expect(useActivityStore.getState().records.some((entry) => entry.id === "live-2")).toBe(false);
    expect(ids(useActivityStore.getState().records)).toEqual(["live-1", "act-page"]);

    unmount();
    expect(listeners.size).toBe(0);
  });
});
