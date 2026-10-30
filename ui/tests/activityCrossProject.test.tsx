import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { act, render, renderHook, screen, waitFor, within } from "./test-utils";
import { SessionPanel } from "../src/components/layout/SessionPanel";
import { mockApplyBoardCommands, mockResetBoard } from "../src/ipc/mock";
import { useActivitySync } from "../src/hooks/useActivitySync";
import { selectUndoEntryId, useActivityStore } from "../src/store/activity";
import { useProjectStore } from "../src/store/project";
import { useUIStore } from "../src/store/uiStore";

const PROJECT_A = "proj-cross-a";
const PROJECT_B = "proj-cross-b";

function project(id: string, name: string) {
  return {
    id,
    name,
    description: "",
    createdAt: "2026-10-30T00:00:00Z",
    updatedAt: "2026-10-30T00:00:00Z",
  };
}

function rowFor(id: string): HTMLElement {
  const row = screen
    .getAllByTestId("activity-row")
    .find((candidate) => candidate.getAttribute("data-activity-id") === id);
  if (!row) throw new Error(`no activity row ${id}`);
  return row;
}

beforeEach(() => {
  mockResetBoard();
  useActivityStore.getState().reset();
  useProjectStore.setState({
    projects: [project(PROJECT_A, "Project A"), project(PROJECT_B, "Project B")],
    activeProjectId: PROJECT_B,
  });
  useUIStore.setState({ sessionPanelOpen: true, canvasTrace: null });
});

afterEach(() => {
  vi.restoreAllMocks();
  mockResetBoard();
  useUIStore.setState({ canvasTrace: null });
});

describe("useActivitySync across projects", () => {
  test("a record added while away leaves its project undoable after switching back", async () => {
    const { rerender } = renderHook(
      ({ projectId }: { projectId: string }) => useActivitySync(projectId),
      { initialProps: { projectId: PROJECT_B } },
    );
    await waitFor(() => expect(useActivityStore.getState().loaded).toBe(true));
    expect(useActivityStore.getState().projectId).toBe(PROJECT_B);

    // A board change lands in A while the shell is still on B: the sync hook
    // must remember A's history even though A's feed is not visible.
    const applied = await act(async () =>
      mockApplyBoardCommands({
        projectId: PROJECT_A,
        commandBatch: {
          id: "batch-away",
          base_version: 0,
          commands: [
            {
              type: "add_item",
              item_id: "note-away",
              kind: { type: "note", content: "Away" },
              position: { x: 0, y: 0 },
              size: { width: 120, height: 80 },
            },
          ],
        },
      }),
    );
    expect(applied.status).toBe("applied");
    const entryId = applied.historyEntryId;
    expect(entryId).not.toBeNull();

    const away = useActivityStore.getState();
    expect(away.projectId).toBe(PROJECT_B);
    expect(away.records).toEqual([]);
    expect(selectUndoEntryId(away)).toBeNull();

    // Back on A, the feed loads its page and restores the live stack.
    rerender({ projectId: PROJECT_A });
    await waitFor(() => {
      const state = useActivityStore.getState();
      expect(state.projectId).toBe(PROJECT_A);
      expect(state.loaded).toBe(true);
    });

    const back = useActivityStore.getState();
    expect(selectUndoEntryId(back)).toBe(entryId);
    const live = back.records.find((record) => record.historyEntryId === entryId);
    expect(live).toBeDefined();

    render(<SessionPanel />);
    expect(within(rowFor(live!.id)).getByTestId("activity-undo")).toBeInTheDocument();
  });
});
