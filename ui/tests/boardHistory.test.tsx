import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { toast } from "sonner";
import { fireEvent, render, screen, waitFor, within } from "./test-utils";
import { SessionPanel } from "../src/components/layout/SessionPanel";
import * as ipc from "../src/ipc";
import {
  mockApplyBoardCommands,
  mockGetBoardSnapshot,
  mockOpInvoke,
  mockRedoBoard,
  mockResetBoard,
  mockUndoBoard,
} from "../src/ipc/mock";
import type { ActivityRecord, BoardCommandResultDto } from "../src/ipc/schemas";
import { runBoardHistory } from "../src/lib/boardHistory";
import { useActivityStore } from "../src/store/activity";
import { useBoardStore } from "../src/store/board";
import { useProjectStore } from "../src/store/project";
import { useUIStore } from "../src/store/uiStore";

vi.mock("sonner", () => ({
  toast: {
    error: vi.fn(),
    success: vi.fn(),
  },
}));

const PROJECT = "proj-history";

async function applyNote(
  baseVersion: number,
  batchId: string,
  itemId: string,
): Promise<BoardCommandResultDto> {
  return mockApplyBoardCommands({
    projectId: PROJECT,
    commandBatch: {
      id: batchId,
      base_version: baseVersion,
      commands: [
        {
          type: "add_item",
          item_id: itemId,
          kind: { type: "note", content: itemId },
          position: { x: 0, y: 0 },
          size: { width: 100, height: 100 },
        },
      ],
    },
  });
}

async function activityRecords(): Promise<ActivityRecord[]> {
  const page = await mockOpInvoke({
    projectId: PROJECT,
    name: "activity.list",
    input: { limit: 50 },
  });
  return (page.output as { records: ActivityRecord[] }).records;
}

function rowFor(id: string): HTMLElement {
  const row = screen
    .getAllByTestId("activity-row")
    .find((candidate) => candidate.getAttribute("data-activity-id") === id);
  if (!row) throw new Error(`no activity row ${id}`);
  return row;
}

/** Route the IPC surface through the in-memory host mirror. */
function useMockHost() {
  const undo = vi
    .spyOn(ipc, "undoBoard")
    .mockImplementation((projectId, expectedEntryId) =>
      mockUndoBoard({ projectId, expectedEntryId: expectedEntryId ?? null }),
    );
  const redo = vi
    .spyOn(ipc, "redoBoard")
    .mockImplementation((projectId, expectedEntryId) =>
      mockRedoBoard({ projectId, expectedEntryId: expectedEntryId ?? null }),
    );
  vi.spyOn(ipc, "opInvoke").mockImplementation((projectId, name, input) =>
    mockOpInvoke({ projectId, name, input }),
  );
  vi.spyOn(ipc, "getBoardSnapshot").mockImplementation((projectId) =>
    mockGetBoardSnapshot({ projectId }),
  );
  return { undo, redo };
}

beforeEach(() => {
  vi.clearAllMocks();
  mockResetBoard();
  useProjectStore.setState({
    projects: [
      {
        id: PROJECT,
        name: "Blog workspace",
        description: "",
        createdAt: "2026-10-30T00:00:00Z",
        updatedAt: "2026-10-30T00:00:00Z",
      },
    ],
    activeProjectId: PROJECT,
  });
  useUIStore.setState({ sessionPanelOpen: true, canvasTrace: null });
  useBoardStore.getState().hydrate({
    id: "",
    projectId: PROJECT,
    version: 0,
    items: [],
  });
  useActivityStore.getState().reset();
  // These tests feed live records directly, so keep the store on the project;
  // only the loaded page is absent.
  useActivityStore.setState({ projectId: PROJECT });
});

afterEach(() => {
  vi.restoreAllMocks();
  useUIStore.setState({ canvasTrace: null });
});

describe("board history guard", () => {
  test("a lost activity event refuses Undo instead of undoing another change", async () => {
    useMockHost();

    const first = await applyNote(0, "batch-1", "note-1");
    const second = await applyNote(1, "batch-2", "note-2");
    const firstId = first.historyEntryId;
    const secondId = second.historyEntryId;
    expect(firstId).toBeTruthy();
    expect(secondId).toBeTruthy();
    expect(secondId).not.toBe(firstId);

    const records = await activityRecords();
    expect(records.map((record) => record.historyEntryId)).toEqual([secondId, firstId]);
    const [latest, older] = records;

    // Only the older record reached the feed; the newer event was lost.
    useActivityStore.getState().append(PROJECT, older);
    expect(useActivityStore.getState().undoStack).toEqual([firstId]);

    const snapshot = await mockGetBoardSnapshot({ projectId: PROJECT });
    useBoardStore.getState().hydrate(snapshot);
    expect(useBoardStore.getState().items.map((item) => item.id)).toEqual(["note-1", "note-2"]);

    render(<SessionPanel />);
    expect(within(rowFor(older.id)).getByTestId("activity-undo")).toBeInTheDocument();

    fireEvent.click(within(rowFor(older.id)).getByTestId("activity-undo"));

    await waitFor(() =>
      expect(toast.error).toHaveBeenCalledWith("The activity feed was out of date", {
        description: "Nothing was undone. The undo and redo targets were resynced.",
      }),
    );

    // The host rejected the stale id, so the board is untouched.
    await waitFor(() =>
      expect(useBoardStore.getState().items.map((item) => item.id)).toEqual(["note-1", "note-2"]),
    );
    expect((await mockGetBoardSnapshot({ projectId: PROJECT })).items).toHaveLength(2);

    // The stacks now follow the host's actual top.
    await waitFor(() => expect(useActivityStore.getState().undoStack).toEqual([secondId]));
    expect(useActivityStore.getState().redoStack).toEqual([]);

    // The forced feed read brought the lost record back, with Undo on it.
    await waitFor(() => expect(screen.getAllByTestId("activity-row")).toHaveLength(2));
    expect(within(rowFor(latest.id)).getByTestId("activity-undo")).toBeInTheDocument();
    expect(within(rowFor(older.id)).queryByTestId("activity-undo")).toBeNull();
  });

  test("a matching expected id lets Undo proceed", async () => {
    const { undo } = useMockHost();

    const first = await applyNote(0, "batch-1", "note-1");
    const second = await applyNote(1, "batch-2", "note-2");
    const records = await activityRecords();
    expect(records.map((record) => record.historyEntryId)).toEqual([
      second.historyEntryId,
      first.historyEntryId,
    ]);

    // Both live events arrived in order.
    useActivityStore.getState().append(PROJECT, records[1]);
    useActivityStore.getState().append(PROJECT, records[0]);
    expect(useActivityStore.getState().undoStack).toEqual([
      first.historyEntryId,
      second.historyEntryId,
    ]);

    useBoardStore.getState().hydrate(await mockGetBoardSnapshot({ projectId: PROJECT }));

    const ok = await runBoardHistory("undo");

    expect(ok).toBe(true);
    expect(undo).toHaveBeenCalledWith(PROJECT, second.historyEntryId);
    expect(toast.error).not.toHaveBeenCalled();
    const snapshot = await mockGetBoardSnapshot({ projectId: PROJECT });
    expect(snapshot.items.map((item) => item.id)).toEqual(["note-1"]);
  });

  test("an unguarded failure keeps the existing undo toast", async () => {
    const { undo } = useMockHost();
    undo.mockRejectedValueOnce(new Error("history unavailable"));

    const first = await applyNote(0, "batch-1", "note-1");
    useActivityStore.getState().append(PROJECT, (await activityRecords())[0]);
    expect(useActivityStore.getState().undoStack).toEqual([first.historyEntryId]);

    expect(await runBoardHistory("undo")).toBe(false);

    expect(undo).toHaveBeenCalledWith(PROJECT, first.historyEntryId);
    expect(toast.error).toHaveBeenCalledWith("Could not undo", {
      description: "history unavailable",
    });
    expect(useActivityStore.getState().undoStack).toEqual([first.historyEntryId]);
  });

  test("an empty stack sends no request and shows no error", async () => {
    const undo = vi.spyOn(ipc, "undoBoard");
    const redo = vi.spyOn(ipc, "redoBoard");

    expect(await runBoardHistory("undo")).toBe(false);
    expect(await runBoardHistory("redo")).toBe(false);

    expect(undo).not.toHaveBeenCalled();
    expect(redo).not.toHaveBeenCalled();
    expect(toast.error).not.toHaveBeenCalled();
  });

  test("a feed loaded for another project is not an undo target", async () => {
    const undo = vi.spyOn(ipc, "undoBoard");
    useActivityStore.setState({ projectId: "proj-other", undoStack: ["h-other"] });

    expect(await runBoardHistory("undo")).toBe(false);

    expect(undo).not.toHaveBeenCalled();
    expect(toast.error).not.toHaveBeenCalled();
  });

  test("concurrent undo calls share one host request", async () => {
    let resolveUndo!: (result: BoardCommandResultDto) => void;
    const hostCall = new Promise<BoardCommandResultDto>((resolve) => {
      resolveUndo = resolve;
    });
    const undo = vi.spyOn(ipc, "undoBoard").mockReturnValue(hostCall);

    const applied = await applyNote(0, "batch-1", "note-1");
    const historyId = applied.historyEntryId;
    expect(historyId).toBeTruthy();
    useActivityStore.getState().append(PROJECT, (await activityRecords())[0]);
    expect(useActivityStore.getState().undoStack).toEqual([historyId]);

    const first = runBoardHistory("undo");
    const second = runBoardHistory("undo");
    expect(undo).toHaveBeenCalledTimes(1);
    expect(undo).toHaveBeenCalledWith(PROJECT, historyId);

    resolveUndo({
      status: "applied",
      version: 2,
      changes: [],
      diagnostics: [],
      historyEntryId: historyId,
    });
    expect(await first).toBe(true);
    expect(await second).toBe(true);
    expect(undo).toHaveBeenCalledTimes(1);

    // Once settled the slot clears, so a later call reaches the host again.
    expect(await runBoardHistory("undo")).toBe(true);
    expect(undo).toHaveBeenCalledTimes(2);
  });

  test("a conflict after a workspace switch leaves the other feed and skips the refresh", async () => {
    const other = "proj-other";
    const applied = await applyNote(0, "batch-1", "note-1");
    const staleId = applied.historyEntryId;
    expect(staleId).toBeTruthy();
    useActivityStore.getState().append(PROJECT, (await activityRecords())[0]);
    expect(useActivityStore.getState().undoStack).toEqual([staleId]);

    let rejectUndo!: (reason: unknown) => void;
    const undo = vi.spyOn(ipc, "undoBoard").mockReturnValue(
      new Promise<BoardCommandResultDto | null>((_resolve, reject) => {
        rejectUndo = reject;
      }),
    );
    const getSnapshot = vi.spyOn(ipc, "getBoardSnapshot");

    const otherRecord: ActivityRecord = {
      id: "act-other",
      at: "2026-10-30T00:00:00Z",
      actor: { kind: "Human", id: null, label: null },
      summary: "Added a note",
      summaryKey: "activity.board.addNote",
      summaryParams: { count: 1 },
      targets: ["note-other"],
      kind: "applied",
      historyEntryId: "h-other",
    };
    vi.spyOn(ipc, "opInvoke").mockImplementation(async (projectId) => ({
      step: {},
      output:
        projectId === other
          ? { records: [otherRecord], has_more: false }
          : { records: [], has_more: false },
    }));

    const pending = runBoardHistory("undo");
    expect(undo).toHaveBeenCalledWith(PROJECT, staleId);

    // The user switches to another project while the request is in flight.
    useProjectStore.setState({ activeProjectId: other });
    await useActivityStore.getState().load(other);
    expect(useActivityStore.getState().projectId).toBe(other);
    expect(useActivityStore.getState().records.map((record) => record.id)).toEqual(["act-other"]);

    rejectUndo({
      code: "conflict",
      message: `board undo expected ${staleId} but the current entry is h-host-top`,
      details: {
        project_id: PROJECT,
        action: "undo",
        expected_entry_id: staleId,
        undo_entry_id: "h-host-top",
        redo_entry_id: null,
      },
    });
    expect(await pending).toBe(false);

    // The other feed stays current; nothing is refreshed and no notice shown.
    const current = useActivityStore.getState();
    expect(current.projectId).toBe(other);
    expect(current.records.map((record) => record.id)).toEqual(["act-other"]);
    expect(current.error).toBeNull();
    expect(getSnapshot).not.toHaveBeenCalled();
    expect(toast.error).not.toHaveBeenCalled();

    // The host's top is remembered for the project that was left.
    await useActivityStore.getState().load(PROJECT);
    expect(useActivityStore.getState().undoStack).toEqual(["h-host-top"]);
  });
});
