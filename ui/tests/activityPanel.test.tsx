import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "./test-utils";
import { SessionPanel } from "../src/components/layout/SessionPanel";
import * as ipc from "../src/ipc";
import type { ActivityRecord } from "../src/ipc/schemas";
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

let sequence = 0;

function record(overrides: Partial<ActivityRecord> = {}): ActivityRecord {
  sequence += 1;
  return {
    id: `act-${sequence}`,
    at: new Date().toISOString(),
    actor: { kind: "Human", id: null, label: null },
    summary: "Added a note",
    summaryKey: "activity.board.addNote",
    summaryParams: { count: 1 },
    targets: ["note-1"],
    kind: "applied",
    historyEntryId: null,
    ...overrides,
  };
}

function seed(
  records: ActivityRecord[],
  stacks: { undoStack?: string[]; redoStack?: string[] } = {},
): void {
  useActivityStore.setState({
    projectId: "default",
    records,
    loaded: true,
    loading: false,
    loadingMore: false,
    error: null,
    hasMore: false,
    undoStack: stacks.undoStack ?? [],
    redoStack: stacks.redoStack ?? [],
  });
}

function rowIds(): Array<string | null> {
  return screen.getAllByTestId("activity-row").map((row) => row.getAttribute("data-activity-id"));
}

function rowFor(id: string): HTMLElement {
  const row = screen
    .getAllByTestId("activity-row")
    .find((candidate) => candidate.getAttribute("data-activity-id") === id);
  if (!row) throw new Error(`no activity row ${id}`);
  return row;
}

beforeEach(() => {
  useProjectStore.setState({
    projects: [
      {
        id: "default",
        name: "Blog workspace",
        description: "",
        createdAt: "2026-10-30T00:00:00Z",
        updatedAt: "2026-10-30T00:00:00Z",
      },
    ],
    activeProjectId: "default",
  });
  useUIStore.setState({ sessionPanelOpen: true, canvasTrace: null });
  useBoardStore
    .getState()
    .hydrate({ id: "board-default", projectId: "default", version: 0, items: [] });
  useActivityStore.getState().reset();
});

afterEach(() => {
  vi.restoreAllMocks();
  // The next test resets the feed itself; doing it here would update a panel
  // that Testing Library has not unmounted yet.
  useUIStore.setState({ canvasTrace: null });
});

describe("activity panel feed", () => {
  test("renders newest first with actor labels and localized summaries", () => {
    const agent = record({
      id: "agent-apply",
      actor: { kind: "Agent", id: "agent-1", label: "Agent" },
      summaryKey: "activity.board.apply",
      summaryParams: { count: 2 },
      targets: ["note-1", "note-2"],
    });
    const note = record({ id: "human-note" });
    const op = record({
      id: "human-op",
      summary: "Ran collection.create",
      summaryKey: "activity.op",
      summaryParams: { op: "collection.create" },
      targets: [],
    });
    const undone = record({
      id: "human-undo",
      kind: "undone",
      summary: "Undid the last board change",
      summaryKey: "activity.board.undo",
      summaryParams: {},
      targets: ["note-1"],
    });
    seed([agent, note, op, undone]);

    render(<SessionPanel />);

    expect(screen.getByRole("heading", { name: "Blog workspace" })).toBeInTheDocument();
    expect(screen.getByText("Workspace activity")).toBeInTheDocument();
    expect(rowIds()).toEqual(["agent-apply", "human-note", "human-op", "human-undo"]);

    expect(within(rowFor("agent-apply")).getByText("Agent")).toBeInTheDocument();
    expect(within(rowFor("agent-apply")).getByText("Applied 2 board changes")).toBeInTheDocument();
    expect(within(rowFor("human-note")).getByText("You")).toBeInTheDocument();
    expect(within(rowFor("human-note")).getByText("Added a note")).toBeInTheDocument();
    expect(within(rowFor("human-op")).getByText("Ran collection.create")).toBeInTheDocument();
    expect(within(rowFor("human-undo")).getByText("Undone")).toBeInTheDocument();
  });

  test("falls back to the host summary for an unknown key", () => {
    seed([record({ id: "custom", summaryKey: "custom.key", summary: "Custom sentence" })]);
    render(<SessionPanel />);
    expect(within(rowFor("custom")).getByText("Custom sentence")).toBeInTheDocument();
  });

  test("filters by actor with pressed chips", () => {
    seed([
      record({ id: "human", actor: { kind: "Human", id: null, label: null } }),
      record({ id: "agent", actor: { kind: "Agent", id: "agent-1", label: "Agent" } }),
      record({ id: "importer", actor: { kind: "Importer", id: null, label: null } }),
    ]);
    render(<SessionPanel />);

    const all = screen.getByTestId("activity-chip-all");
    const you = screen.getByTestId("activity-chip-you");
    const agent = screen.getByTestId("activity-chip-agent");
    expect(all).toHaveAttribute("aria-pressed", "true");

    fireEvent.click(you);
    expect(you).toHaveAttribute("aria-pressed", "true");
    expect(all).toHaveAttribute("aria-pressed", "false");
    expect(rowIds()).toEqual(["human"]);

    fireEvent.click(agent);
    expect(rowIds()).toEqual(["agent"]);

    fireEvent.click(all);
    expect(rowIds()).toEqual(["human", "agent", "importer"]);
  });

  test("shows undo and redo only on the newest matching records", () => {
    seed(
      [
        record({ id: "newest", kind: "applied", historyEntryId: "h-2" }),
        record({ id: "older", kind: "redone", historyEntryId: "h-2" }),
        record({
          id: "undone",
          kind: "undone",
          historyEntryId: "h-3",
          summaryKey: "activity.board.undo",
          summaryParams: {},
        }),
        record({ id: "op", summaryKey: "activity.op", summaryParams: { op: "asset.import" } }),
      ],
      { undoStack: ["h-2"], redoStack: ["h-3"] },
    );

    render(<SessionPanel />);

    expect(screen.getAllByTestId("activity-undo")).toHaveLength(1);
    expect(within(rowFor("newest")).getByTestId("activity-undo")).toBeInTheDocument();
    expect(within(rowFor("older")).queryByTestId("activity-undo")).toBeNull();

    expect(screen.getAllByTestId("activity-redo")).toHaveLength(1);
    expect(within(rowFor("undone")).getByTestId("activity-redo")).toBeInTheDocument();

    expect(
      within(rowFor("newest")).getByRole("button", { name: "Undo: Added a note" }),
    ).toBeInTheDocument();
  });

  test("clicking undo and redo calls the board store actions", async () => {
    const undoSpy = vi.spyOn(ipc, "undoBoard").mockResolvedValue({
      status: "applied",
      version: 1,
      changes: [],
      diagnostics: [],
      historyEntryId: "h-2",
    });
    const redoSpy = vi.spyOn(ipc, "redoBoard").mockResolvedValue({
      status: "applied",
      version: 2,
      changes: [],
      diagnostics: [],
      historyEntryId: "h-3",
    });

    seed(
      [
        record({ id: "applied", historyEntryId: "h-2" }),
        record({ id: "undone", kind: "undone", historyEntryId: "h-3" }),
      ],
      { undoStack: ["h-2"], redoStack: ["h-3"] },
    );

    render(<SessionPanel />);

    fireEvent.click(within(rowFor("applied")).getByTestId("activity-undo"));
    await waitFor(() => expect(undoSpy).toHaveBeenCalledWith("default"));

    fireEvent.click(within(rowFor("undone")).getByTestId("activity-redo"));
    await waitFor(() => expect(redoSpy).toHaveBeenCalledWith("default"));
  });

  test("a failed undo is toasted", async () => {
    const { toast } = await import("sonner");
    vi.spyOn(ipc, "undoBoard").mockRejectedValueOnce(new Error("history unavailable"));

    seed([record({ id: "applied", historyEntryId: "h-2" })], { undoStack: ["h-2"] });
    render(<SessionPanel />);

    fireEvent.click(within(rowFor("applied")).getByTestId("activity-undo"));

    await waitFor(() =>
      expect(toast.error).toHaveBeenCalledWith("Could not undo", {
        description: "history unavailable",
      }),
    );
  });

  test("hovering a record traces its targets and leaving clears the trace", () => {
    const target = record({ id: "target", targets: ["note-1", "note-2"] });
    const op = record({
      id: "op",
      summaryKey: "activity.op",
      summaryParams: { op: "asset.import" },
      targets: [],
    });
    seed([target, op]);
    render(<SessionPanel />);

    fireEvent.pointerOver(rowFor("target"));
    expect(useUIStore.getState().canvasTrace).toEqual({
      targets: ["note-1", "note-2"],
      actorKind: "Human",
    });

    fireEvent.pointerOut(rowFor("target"));
    expect(useUIStore.getState().canvasTrace).toBeNull();

    // Records without targets never start a trace.
    fireEvent.pointerOver(rowFor("op"));
    expect(useUIStore.getState().canvasTrace).toBeNull();
  });

  test("agent records trace with the agent actor kind", () => {
    seed([
      record({
        id: "agent",
        actor: { kind: "Agent", id: "agent-1", label: "Agent" },
        targets: ["note-3"],
      }),
    ]);
    render(<SessionPanel />);

    fireEvent.pointerOver(rowFor("agent"));
    expect(useUIStore.getState().canvasTrace).toEqual({
      targets: ["note-3"],
      actorKind: "Agent",
    });
  });

  test("closing and unmounting clear a hover trace", () => {
    seed([record({ id: "target" })]);
    const { unmount } = render(<SessionPanel />);

    fireEvent.pointerOver(rowFor("target"));
    expect(useUIStore.getState().canvasTrace).not.toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Close panel" }));
    expect(useUIStore.getState().canvasTrace).toBeNull();

    act(() => {
      useUIStore.setState({ sessionPanelOpen: true });
      seed([record({ id: "again" })]);
    });
    fireEvent.pointerOver(rowFor("again"));
    expect(useUIStore.getState().canvasTrace).not.toBeNull();

    unmount();
    expect(useUIStore.getState().canvasTrace).toBeNull();
  });
});

describe("activity panel states", () => {
  test("shows the loading and empty states", () => {
    useActivityStore.setState({ loading: true, records: [] });
    const { unmount } = render(<SessionPanel />);
    expect(screen.getByTestId("activity-loading")).toBeInTheDocument();
    unmount();

    useActivityStore.setState({ loading: false, records: [], loaded: true });
    render(<SessionPanel />);
    expect(screen.getByTestId("activity-empty")).toBeInTheDocument();
    expect(screen.getByText("No activity yet")).toBeInTheDocument();
  });

  test("a failed first load offers a retry", async () => {
    const opInvoke = vi
      .spyOn(ipc, "opInvoke")
      .mockResolvedValue({ step: {}, output: { records: [], has_more: false } });
    useActivityStore.setState({ error: "host down", records: [], loaded: false });

    render(<SessionPanel />);
    expect(screen.getByTestId("activity-error")).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("activity-retry"));

    await waitFor(() =>
      expect(opInvoke).toHaveBeenCalledWith("default", "activity.list", { limit: 50 }),
    );
  });

  test("loads an older page from the bottom of the list", async () => {
    const opInvoke = vi
      .spyOn(ipc, "opInvoke")
      .mockResolvedValue({ step: {}, output: { records: [], has_more: false } });
    seed([record({ id: "only" })]);
    useActivityStore.setState({ hasMore: true });

    render(<SessionPanel />);
    fireEvent.click(screen.getByTestId("activity-load-more"));

    await waitFor(() =>
      expect(opInvoke).toHaveBeenCalledWith("default", "activity.list", {
        limit: 50,
        before: "only",
      }),
    );
  });

  test("an empty filter result keeps the panel copy", () => {
    seed([record({ id: "human", actor: { kind: "Human", id: null, label: null } })]);
    render(<SessionPanel />);

    fireEvent.click(screen.getByTestId("activity-chip-agent"));
    expect(screen.getByTestId("activity-no-matches")).toBeInTheDocument();
    expect(screen.getByText("No activity matches this filter.")).toBeInTheDocument();
  });
});
