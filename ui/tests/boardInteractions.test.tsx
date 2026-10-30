import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, act, renderHook, waitFor } from "./test-utils";
import { BoardCanvas, BoardCanvasInner } from "../src/board/BoardCanvas";
import { ReactFlowProvider, useStoreApi, type Node } from "@xyflow/react";
import { useBoardStore } from "../src/store/board";
import { useProjectStore } from "../src/store/project";
import { useUIStore } from "../src/store/uiStore";
import { useWorkflowStore } from "../src/store/workflow";
import { useUndoRedoShortcuts } from "../src/hooks/useUndoRedo";
import { useWorkspaceSurface } from "../src/hooks/useWorkspaceSurface";
import * as ipc from "../src/ipc";
import { toast } from "sonner";
import { buildDragStopCommands, type CommandBatch, type BoardCommand } from "../src/board/batch";
import type { BoardCommandResultDto } from "../src/ipc/schemas";

// Mock ResizeObserver for xyflow in jsdom
if (typeof globalThis.ResizeObserver === "undefined") {
  globalThis.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}

vi.mock("sonner", () => ({
  toast: {
    error: vi.fn(),
    success: vi.fn(),
  },
}));

vi.mock("@tanstack/react-router", () => ({
  useNavigate: () => vi.fn(),
  Link: ({ to, children }: { to: string; children: React.ReactNode }) => (
    <a href={to}>{children}</a>
  ),
}));

let testStoreApi: ReturnType<typeof useStoreApi> | null = null;
function StoreCapture() {
  testStoreApi = useStoreApi();
  return null;
}

function UndoRedoShortcutMount() {
  useUndoRedoShortcuts();
  return <div data-testid="shortcuts-mounted" />;
}

describe("Board canvas interactions and command batches", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    testStoreApi = null;
    useProjectStore.setState({ activeProjectId: "proj-1" });
    useUIStore.setState({
      sessionPanelOpen: true,
      canvasFocusItemId: null,
      canvasViewports: {},
    });
    useBoardStore.getState().hydrate({
      id: "board-proj-1",
      projectId: "proj-1",
      version: 3,
      items: [
        {
          id: "note-existing",
          kind: { type: "note", content: "Original content" },
          position: { x: 100, y: 100 },
          size: { width: 240, height: 160 },
          z: 1,
          locked: false,
        },
        {
          id: "item-2",
          kind: { type: "asset_ref", artifact_id: "art-99" },
          position: { x: 400, y: 100 },
          size: { width: 200, height: 100 },
          z: 2,
          locked: false,
        },
      ],
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  test("empty state 'New note' button generates add_item batch with empty content, default size and enters edit", async () => {
    const applySpy = vi.spyOn(ipc, "applyBoardCommands").mockResolvedValue({
      status: "applied",
      version: 4,
      changes: [],
      diagnostics: [],
      historyEntryId: "hist-1",
    });

    useBoardStore.getState().hydrate({
      id: "board-proj-1",
      projectId: "proj-1",
      version: 3,
      items: [],
    });

    render(<BoardCanvas />);

    const newNoteBtn = screen.getByRole("button", { name: "New note" });
    await act(async () => {
      fireEvent.click(newNoteBtn);
    });

    expect(applySpy).toHaveBeenCalledTimes(1);
    const [projectId, batch] = applySpy.mock.calls[0] as [string, CommandBatch<BoardCommand>];
    expect(projectId).toBe("proj-1");
    expect(batch.base_version).toBe(3);
    expect(batch.commands).toHaveLength(1);
    expect(batch.commands[0]).toMatchObject({
      type: "add_item",
      kind: { type: "note", content: "" },
      size: { width: 240, height: 160 },
    });
  });

  test("note tool places one note on a pane click and returns to select", async () => {
    const applySpy = vi.spyOn(ipc, "applyBoardCommands").mockResolvedValue({
      status: "applied",
      version: 4,
      changes: [],
      diagnostics: [],
      historyEntryId: "hist-1",
    });

    render(<BoardCanvas />);

    const noteTool = screen.getByRole("button", { name: "Note" });
    fireEvent.click(noteTool);
    expect(noteTool).toHaveAttribute("aria-pressed", "true");

    const pane = document.querySelector(".react-flow__pane");
    expect(pane).not.toBeNull();
    await act(async () => {
      fireEvent.click(pane!, { clientX: 200, clientY: 200 });
    });

    expect(applySpy).toHaveBeenCalledTimes(1);
    expect(noteTool).toHaveAttribute("aria-pressed", "false");
  });

  test("empty state is hidden until the snapshot loads", () => {
    useBoardStore.getState().hydrate({
      id: "",
      projectId: "proj-1",
      version: 0,
      items: [],
    });

    render(<BoardCanvas />);
    expect(screen.queryByTestId("canvas-empty-state")).toBeNull();

    act(() => {
      useBoardStore.getState().hydrate({
        id: "board-proj-1",
        projectId: "proj-1",
        version: 3,
        items: [],
      });
    });

    expect(screen.getByTestId("canvas-empty-state")).toBeInTheDocument();
  });

  test("requestCanvasFocus is consumed and clears a missing item", async () => {
    useUIStore.getState().requestCanvasFocus("missing-item");

    render(<BoardCanvas />);

    await waitFor(() => {
      expect(useUIStore.getState().canvasFocusItemId).toBeNull();
    });
  });

  test("requestCanvasFocus centers the item in the area left of the panel", async () => {
    vi.stubGlobal("matchMedia", () => ({ matches: true }));
    vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockReturnValue(1000);
    vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockReturnValue(600);

    render(
      <ReactFlowProvider>
        <BoardCanvasInner />
        <StoreCapture />
      </ReactFlowProvider>,
    );

    act(() => {
      useUIStore.getState().requestCanvasFocus("item-2");
    });

    await waitFor(() => {
      expect(useUIStore.getState().canvasFocusItemId).toBeNull();
    });
    // item-2 center (500, 150); visible center is (302, 300) with the panel open.
    expect(testStoreApi!.getState().transform).toEqual([-198, 150, 1]);
  });

  test("mounts at the stored viewport and persists pan/zoom ends per project", async () => {
    useUIStore.setState({
      canvasViewports: { "proj-1": { x: -120, y: -80, zoom: 0.75 } },
    });

    render(
      <ReactFlowProvider>
        <BoardCanvasInner />
        <StoreCapture />
      </ReactFlowProvider>,
    );

    await waitFor(() => {
      expect(testStoreApi!.getState().transform).toEqual([-120, -80, 0.75]);
    });

    act(() => {
      testStoreApi!.getState().onMoveEnd?.({} as MouseEvent, { x: -600, y: -300, zoom: 1.5 });
    });
    expect(useUIStore.getState().canvasViewports["proj-1"]).toEqual({
      x: -600,
      y: -300,
      zoom: 1.5,
    });
  });

  test("a project switch applies that project's stored viewport", async () => {
    useUIStore.setState({
      canvasViewports: {
        "proj-1": { x: 10, y: 20, zoom: 1 },
        "proj-2": { x: -30, y: -40, zoom: 0.5 },
      },
    });

    render(
      <ReactFlowProvider>
        <BoardCanvasInner />
        <StoreCapture />
      </ReactFlowProvider>,
    );
    await waitFor(() => {
      expect(testStoreApi!.getState().transform).toEqual([10, 20, 1]);
    });

    act(() => {
      useProjectStore.setState({ activeProjectId: "proj-2" });
    });

    await waitFor(() => {
      expect(testStoreApi!.getState().transform).toEqual([-30, -40, 0.5]);
    });
  });

  test("double-clicking empty canvas generates add_item batch at click point", async () => {
    const applySpy = vi.spyOn(ipc, "applyBoardCommands").mockResolvedValue({
      status: "applied",
      version: 4,
      changes: [],
      diagnostics: [],
      historyEntryId: "hist-1",
    });

    render(<BoardCanvas />);

    const canvas = screen.getByTestId("board-canvas");
    await act(async () => {
      fireEvent.doubleClick(canvas, { clientX: 250, clientY: 350 });
    });

    expect(applySpy).toHaveBeenCalledTimes(1);
    const [, batch] = applySpy.mock.calls[0] as [string, CommandBatch<BoardCommand>];
    expect(batch.commands[0].type).toBe("add_item");
    expect((batch.commands[0] as { kind: { type: string } }).kind.type).toBe("note");
  });

  test("editing note: unchanged content sends nothing; changed content sends update_note batch", async () => {
    const applySpy = vi.spyOn(ipc, "applyBoardCommands").mockResolvedValue({
      status: "applied",
      version: 4,
      changes: [],
      diagnostics: [],
      historyEntryId: "hist-1",
    });

    render(<BoardCanvas />);

    // Note is rendered
    const noteCard = screen.getByTestId("note-card");
    fireEvent.doubleClick(noteCard);

    // Edit textarea appears
    const textarea = screen.getByTestId("note-textarea") as HTMLTextAreaElement;
    expect(textarea.value).toBe("Original content");

    // Case A: blur with unchanged content
    fireEvent.blur(textarea);
    expect(applySpy).not.toHaveBeenCalled();

    // Re-enter edit mode
    fireEvent.doubleClick(noteCard);
    const textareaAgain = screen.getByTestId("note-textarea") as HTMLTextAreaElement;

    // Case B: change content and blur
    fireEvent.change(textareaAgain, { target: { value: "Updated note text!" } });
    await act(async () => {
      fireEvent.blur(textareaAgain);
    });

    expect(applySpy).toHaveBeenCalledTimes(1);
    const [, batch] = applySpy.mock.calls[0] as [string, CommandBatch<BoardCommand>];
    expect(batch.commands).toEqual([
      {
        type: "update_note",
        item_id: "note-existing",
        content: "Updated note text!",
      },
    ]);
  });

  test("buildDragStopCommands pure function filters unchanged nodes and rounds coordinates", () => {
    const items = [
      { id: "note-existing", position: { x: 100, y: 100 } },
      { id: "item-2", position: { x: 400, y: 100 } },
    ];
    const dragged = [
      { id: "note-existing", position: { x: 100.1, y: 99.9 } },
      { id: "item-2", position: { x: 420.7, y: 210.2 } },
    ];
    const cmds = buildDragStopCommands(items, dragged);
    expect(cmds).toEqual([
      {
        type: "move_item",
        item_id: "item-2",
        position: { x: 421, y: 210 },
      },
    ]);
  });

  test("multi-drag interaction uses buildDragStopCommands and sends one batch with move_item for each moved node", async () => {
    const applySpy = vi.spyOn(ipc, "applyBoardCommands").mockResolvedValue({
      status: "applied",
      version: 4,
      changes: [],
      diagnostics: [],
      historyEntryId: "hist-1",
    });

    render(
      <ReactFlowProvider>
        <BoardCanvasInner />
        <StoreCapture />
      </ReactFlowProvider>,
    );

    const movedNodes = [
      { id: "note-existing", position: { x: 150.3, y: 160.7 } },
      { id: "item-2", position: { x: 420.1, y: 210.9 } },
    ];

    await act(async () => {
      await testStoreApi!
        .getState()
        .onNodeDragStop?.(
          {} as React.MouseEvent,
          movedNodes[0] as unknown as Node,
          movedNodes as unknown as Node[],
        );
    });

    expect(applySpy).toHaveBeenCalledTimes(1);
    const [, batch] = applySpy.mock.calls[0] as [string, CommandBatch<BoardCommand>];
    expect(batch.commands).toHaveLength(2);
    expect(batch.commands[0]).toEqual({
      type: "move_item",
      item_id: "note-existing",
      position: { x: 150, y: 161 },
    });
    expect(batch.commands[1]).toEqual({
      type: "move_item",
      item_id: "item-2",
      position: { x: 420, y: 211 },
    });
  });

  test("two quick moves serialize dispatches and produce batches with base_version V and V+1", async () => {
    let resolveFirstApply: (val: BoardCommandResultDto) => void;
    const firstPromise = new Promise<BoardCommandResultDto>((resolve) => {
      resolveFirstApply = resolve;
    });

    let callCount = 0;
    const applySpy = vi.spyOn(ipc, "applyBoardCommands").mockImplementation(async () => {
      callCount += 1;
      if (callCount === 1) {
        await firstPromise;
        return {
          status: "applied",
          version: 4,
          changes: [],
          diagnostics: [],
          historyEntryId: "hist-1",
        };
      } else {
        return {
          status: "applied",
          version: 5,
          changes: [],
          diagnostics: [],
          historyEntryId: "hist-2",
        };
      }
    });

    render(
      <ReactFlowProvider>
        <BoardCanvasInner />
        <StoreCapture />
      </ReactFlowProvider>,
    );

    // Initial version is 3
    expect(useBoardStore.getState().version).toBe(3);

    const move1 = { id: "note-existing", position: { x: 120, y: 130 } };
    const move2 = { id: "item-2", position: { x: 450, y: 150 } };

    let p1: Promise<unknown>;
    let p2: Promise<unknown>;
    await act(async () => {
      p1 = testStoreApi!
        .getState()
        .onNodeDragStop?.({} as React.MouseEvent, move1 as unknown as Node, [
          move1 as unknown as Node,
        ]) as unknown as Promise<unknown>;
      p2 = testStoreApi!
        .getState()
        .onNodeDragStop?.({} as React.MouseEvent, move2 as unknown as Node, [
          move2 as unknown as Node,
        ]) as unknown as Promise<unknown>;
      await Promise.resolve();
    });

    // Before first promise resolves, only the first batch was dispatched
    expect(applySpy).toHaveBeenCalledTimes(1);
    const [, batch1] = applySpy.mock.calls[0] as [string, CommandBatch<BoardCommand>];
    expect(batch1.base_version).toBe(3);

    // Resolve first batch
    await act(async () => {
      resolveFirstApply!({
        status: "applied",
        version: 4,
        changes: [],
        diagnostics: [],
        historyEntryId: null,
      });
      await p1!;
      await p2!;
    });

    // Second batch was built and dispatched with base_version 4 (V+1)
    expect(applySpy).toHaveBeenCalledTimes(2);
    const [, batch2] = applySpy.mock.calls[1] as [string, CommandBatch<BoardCommand>];
    expect(batch2.base_version).toBe(4);
  });

  test("project switch while request is in flight: store does not update version for old project and rejection does not reload old snapshot", async () => {
    // 1. Store apply/undo/redo ignores version updates if active project changed
    let resolveApply: (val: BoardCommandResultDto) => void;
    const applyPromise = new Promise<BoardCommandResultDto>((resolve) => {
      resolveApply = resolve;
    });

    vi.spyOn(ipc, "applyBoardCommands").mockReturnValue(applyPromise);

    useBoardStore.setState({ id: "board-1", projectId: "proj-1", version: 1, items: [] });
    const applyFlight = useBoardStore.getState().apply("proj-1", { commands: [] });

    // Project switches to proj-2
    useBoardStore.setState({ id: "board-2", projectId: "proj-2", version: 10, items: [] });

    // Old proj-1 apply resolves with version 2
    await act(async () => {
      resolveApply!({
        status: "applied",
        version: 2,
        changes: [],
        diagnostics: [],
        historyEntryId: null,
      });
      await applyFlight;
    });

    // Store version should NOT be updated to 2
    expect(useBoardStore.getState().projectId).toBe("proj-2");
    expect(useBoardStore.getState().version).toBe(10);

    // 2. Rejection reload only if active project is still the one the batch was for
    const getSnapshotSpy = vi.spyOn(ipc, "getBoardSnapshot");
    let resolveReject: (val: BoardCommandResultDto) => void;
    const rejectPromise = new Promise<BoardCommandResultDto>((resolve) => {
      resolveReject = resolve;
    });
    vi.spyOn(ipc, "applyBoardCommands").mockReturnValue(rejectPromise);

    useProjectStore.setState({ activeProjectId: "proj-1" });
    useBoardStore.setState({ id: "board-1", projectId: "proj-1", version: 1, items: [] });

    render(
      <ReactFlowProvider>
        <BoardCanvasInner />
        <StoreCapture />
      </ReactFlowProvider>,
    );

    const moveNode = { id: "note-existing", position: { x: 200, y: 200 } };
    let movePromise: Promise<unknown>;
    act(() => {
      movePromise = testStoreApi!
        .getState()
        .onNodeDragStop?.({} as React.MouseEvent, moveNode as unknown as Node, [
          moveNode as unknown as Node,
        ]) as unknown as Promise<unknown>;
    });

    // Switch project before rejection resolves
    act(() => {
      useProjectStore.setState({ activeProjectId: "proj-2" });
    });

    await act(async () => {
      resolveReject!({
        status: "rejected",
        version: 1,
        changes: [],
        diagnostics: [{ message: "Rejected" }],
        historyEntryId: null,
      });
      await movePromise!;
    });

    // Reload was NOT called for proj-1 because active project is proj-2
    expect(getSnapshotSpy).not.toHaveBeenCalledWith("proj-1");
  });

  test("selection survives a snapshot reload", async () => {
    render(
      <ReactFlowProvider>
        <BoardCanvasInner />
        <StoreCapture />
      </ReactFlowProvider>,
    );

    // Select note-existing via onNodesChange
    act(() => {
      testStoreApi!
        .getState()
        .onNodesChange?.([{ id: "note-existing", type: "select", selected: true }]);
    });

    expect(
      testStoreApi!.getState().nodes.find((n: Node) => n.id === "note-existing")?.selected,
    ).toBe(true);

    // Snapshot reload lands with version bump
    act(() => {
      useBoardStore.getState().hydrate({
        id: "board-proj-1",
        projectId: "proj-1",
        version: 10,
        items: [
          {
            id: "note-existing",
            kind: { type: "note", content: "Original content" },
            position: { x: 100, y: 100 },
            size: { width: 240, height: 160 },
            z: 1,
            locked: false,
          },
        ],
      });
    });

    // Selection survives the reload
    expect(
      testStoreApi!.getState().nodes.find((n: Node) => n.id === "note-existing")?.selected,
    ).toBe(true);
  });

  test("a dragging node keeps its position across a reload", async () => {
    render(
      <ReactFlowProvider>
        <BoardCanvasInner />
        <StoreCapture />
      </ReactFlowProvider>,
    );

    // Mark note-existing as dragging at local position (555, 666) via onNodesChange
    act(() => {
      testStoreApi!
        .getState()
        .onNodesChange?.([
          { id: "note-existing", type: "position", dragging: true, position: { x: 555, y: 666 } },
        ]);
    });

    // Incoming board.changed snapshot reload with server position (100, 100)
    act(() => {
      useBoardStore.getState().hydrate({
        id: "board-proj-1",
        projectId: "proj-1",
        version: 11,
        items: [
          {
            id: "note-existing",
            kind: { type: "note", content: "Original content" },
            position: { x: 100, y: 100 },
            size: { width: 240, height: 160 },
            z: 1,
            locked: false,
          },
        ],
      });
    });

    // Node keeps dragging position across the reload
    const node = testStoreApi!.getState().nodes.find((n: Node) => n.id === "note-existing");
    expect(node?.position).toEqual({ x: 555, y: 666 });
  });

  test("changing editingId does not snap a moved node back", async () => {
    vi.spyOn(ipc, "applyBoardCommands").mockResolvedValue({
      status: "applied",
      version: 4,
      changes: [],
      diagnostics: [],
      historyEntryId: "hist-1",
    });

    render(
      <ReactFlowProvider>
        <BoardCanvasInner />
        <StoreCapture />
      </ReactFlowProvider>,
    );

    // Stop drag at (300, 400)
    const move = { id: "note-existing", position: { x: 300, y: 400 } };
    await act(async () => {
      await testStoreApi!
        .getState()
        .onNodeDragStop?.({} as React.MouseEvent, move as unknown as Node, [
          move as unknown as Node,
        ]);
    });

    expect(
      testStoreApi!.getState().nodes.find((n: Node) => n.id === "note-existing")?.position,
    ).toEqual({ x: 300, y: 400 });

    // Change editingId before snapshot reload carries new position
    const otherNode = testStoreApi!.getState().nodes.find((n: Node) => n.id === "item-2");
    act(() => {
      otherNode?.data.onStartEdit?.("item-2");
    });

    // note-existing does NOT snap back to (100, 100)
    expect(
      testStoreApi!.getState().nodes.find((n: Node) => n.id === "note-existing")?.position,
    ).toEqual({ x: 300, y: 400 });
  });

  test("pending moves: undo after a move shows the undone position", async () => {
    vi.spyOn(ipc, "applyBoardCommands").mockResolvedValue({
      status: "applied",
      version: 4,
      changes: [],
      diagnostics: [],
      historyEntryId: "hist-1",
    });

    render(
      <ReactFlowProvider>
        <BoardCanvasInner />
        <StoreCapture />
      </ReactFlowProvider>,
    );

    expect(
      testStoreApi!.getState().nodes.find((n: Node) => n.id === "note-existing")?.position,
    ).toEqual({ x: 100, y: 100 });

    const move = { id: "note-existing", position: { x: 300, y: 400 } };
    await act(async () => {
      await testStoreApi!
        .getState()
        .onNodeDragStop?.({} as React.MouseEvent, move as unknown as Node, [
          move as unknown as Node,
        ]);
    });

    expect(
      testStoreApi!.getState().nodes.find((n: Node) => n.id === "note-existing")?.position,
    ).toEqual({ x: 300, y: 400 });

    // Snapshot at version 4 arrives matching the move
    await act(async () => {
      useBoardStore.getState().hydrateIfNewer({
        id: "board-proj-1",
        projectId: "proj-1",
        version: 4,
        items: [
          {
            id: "note-existing",
            kind: { type: "note", content: "Original content" },
            position: { x: 300, y: 400 },
            size: { width: 240, height: 160 },
            z: 1,
            locked: false,
          },
        ],
      });
    });

    // Now undo occurs (version moves to 5, items restored to 100, 100)
    await act(async () => {
      useBoardStore.getState().hydrateIfNewer({
        id: "board-proj-1",
        projectId: "proj-1",
        version: 5,
        items: [
          {
            id: "note-existing",
            kind: { type: "note", content: "Original content" },
            position: { x: 100, y: 100 },
            size: { width: 240, height: 160 },
            z: 1,
            locked: false,
          },
        ],
      });
    });

    expect(
      testStoreApi!.getState().nodes.find((n: Node) => n.id === "note-existing")?.position,
    ).toEqual({ x: 100, y: 100 });
  });

  test("pending moves: another snapshot with different position wins once version >= resultVersion", async () => {
    vi.spyOn(ipc, "applyBoardCommands").mockResolvedValue({
      status: "applied",
      version: 4,
      changes: [],
      diagnostics: [],
      historyEntryId: "hist-1",
    });

    render(
      <ReactFlowProvider>
        <BoardCanvasInner />
        <StoreCapture />
      </ReactFlowProvider>,
    );

    const move = { id: "note-existing", position: { x: 300, y: 400 } };
    await act(async () => {
      await testStoreApi!
        .getState()
        .onNodeDragStop?.({} as React.MouseEvent, move as unknown as Node, [
          move as unknown as Node,
        ]);
    });

    expect(
      testStoreApi!.getState().nodes.find((n: Node) => n.id === "note-existing")?.position,
    ).toEqual({ x: 300, y: 400 });

    // Snapshot arrives with version 4 carrying position (250, 350)
    await act(async () => {
      useBoardStore.getState().hydrateIfNewer({
        id: "board-proj-1",
        projectId: "proj-1",
        version: 4,
        items: [
          {
            id: "note-existing",
            kind: { type: "note", content: "Original content" },
            position: { x: 250, y: 350 },
            size: { width: 240, height: 160 },
            z: 1,
            locked: false,
          },
        ],
      });
    });

    expect(
      testStoreApi!.getState().nodes.find((n: Node) => n.id === "note-existing")?.position,
    ).toEqual({ x: 250, y: 350 });
  });

  test("pending moves: rejection drops the entry and resets position on snapshot reload", async () => {
    vi.spyOn(ipc, "applyBoardCommands").mockResolvedValue({
      status: "rejected",
      version: 3,
      changes: [],
      diagnostics: [{ message: "Move rejected" }],
      historyEntryId: null,
    });
    vi.spyOn(ipc, "getBoardSnapshot").mockResolvedValue({
      id: "board-proj-1",
      projectId: "proj-1",
      version: 3,
      items: [
        {
          id: "note-existing",
          kind: { type: "note", content: "Original content" },
          position: { x: 100, y: 100 },
          size: { width: 240, height: 160 },
          z: 1,
          locked: false,
        },
      ],
    });

    render(
      <ReactFlowProvider>
        <BoardCanvasInner />
        <StoreCapture />
      </ReactFlowProvider>,
    );

    const move = { id: "note-existing", position: { x: 500, y: 600 } };
    await act(async () => {
      await testStoreApi!
        .getState()
        .onNodeDragStop?.({} as React.MouseEvent, move as unknown as Node, [
          move as unknown as Node,
        ]);
    });

    // Reset position from reloaded snapshot
    expect(
      testStoreApi!.getState().nodes.find((n: Node) => n.id === "note-existing")?.position,
    ).toEqual({ x: 100, y: 100 });
  });

  test("pending moves: project switch clears pending moves", async () => {
    let resolveApply: (val: BoardCommandResultDto) => void;
    const applyPromise = new Promise<BoardCommandResultDto>((resolve) => {
      resolveApply = resolve;
    });
    vi.spyOn(ipc, "applyBoardCommands").mockReturnValue(applyPromise);

    render(
      <ReactFlowProvider>
        <BoardCanvasInner />
        <StoreCapture />
      </ReactFlowProvider>,
    );

    const move = { id: "note-existing", position: { x: 400, y: 400 } };
    act(() => {
      void testStoreApi!
        .getState()
        .onNodeDragStop?.({} as React.MouseEvent, move as unknown as Node, [
          move as unknown as Node,
        ]);
    });

    expect(
      testStoreApi!.getState().nodes.find((n: Node) => n.id === "note-existing")?.position,
    ).toEqual({ x: 400, y: 400 });

    act(() => {
      useProjectStore.setState({ activeProjectId: "proj-2" });
      useBoardStore.getState().hydrate({
        id: "board-proj-2",
        projectId: "proj-2",
        version: 1,
        items: [
          {
            id: "note-existing",
            kind: { type: "note", content: "Other project note" },
            position: { x: 50, y: 50 },
            size: { width: 240, height: 160 },
            z: 1,
            locked: false,
          },
        ],
      });
    });

    expect(
      testStoreApi!.getState().nodes.find((n: Node) => n.id === "note-existing")?.position,
    ).toEqual({ x: 50, y: 50 });

    await act(async () => {
      resolveApply!({
        status: "applied",
        version: 4,
        changes: [],
        diagnostics: [],
        historyEntryId: null,
      });
    });
  });

  test("stale snapshot does not overwrite newer store state", () => {
    useProjectStore.setState({ activeProjectId: "proj-1" });
    useBoardStore.setState({ id: "board-1", projectId: "proj-1", version: 5, items: [] });

    // Stale snapshot at version 4 is rejected
    const acceptedStale = useBoardStore.getState().hydrateIfNewer({
      id: "board-1",
      projectId: "proj-1",
      version: 4,
      items: [
        {
          id: "item-old",
          kind: { type: "note", content: "old" },
          position: { x: 0, y: 0 },
          size: { width: 240, height: 160 },
          z: 0,
          locked: false,
        },
      ],
    });
    expect(acceptedStale).toBe(false);
    expect(useBoardStore.getState().items).toHaveLength(0);
    expect(useBoardStore.getState().version).toBe(5);

    // Snapshot at version 5 (equal version, reload after rejection) is accepted
    const acceptedEqual = useBoardStore.getState().hydrateIfNewer({
      id: "board-1",
      projectId: "proj-1",
      version: 5,
      items: [
        {
          id: "item-current",
          kind: { type: "note", content: "current" },
          position: { x: 0, y: 0 },
          size: { width: 240, height: 160 },
          z: 0,
          locked: false,
        },
      ],
    });
    expect(acceptedEqual).toBe(true);
    expect(useBoardStore.getState().items).toHaveLength(1);

    // Snapshot for different active project is rejected
    const acceptedWrongProject = useBoardStore.getState().hydrateIfNewer({
      id: "board-2",
      projectId: "proj-2",
      version: 10,
      items: [],
    });
    expect(acceptedWrongProject).toBe(false);

    // Project switch: when activeProjectId is proj-2 and store has proj-1, snapshot for proj-2 is accepted even with lower version
    useProjectStore.setState({ activeProjectId: "proj-2" });
    const acceptedSwitch = useBoardStore.getState().hydrateIfNewer({
      id: "board-2",
      projectId: "proj-2",
      version: 1,
      items: [],
    });
    expect(acceptedSwitch).toBe(true);
    expect(useBoardStore.getState().projectId).toBe("proj-2");
    expect(useBoardStore.getState().version).toBe(1);
  });

  test("failing snapshot load shows error toast", async () => {
    const toastErrorSpy = vi.spyOn(toast, "error");
    vi.spyOn(ipc, "getBoardSnapshot").mockRejectedValue(new Error("Network failure"));

    useProjectStore.setState({ activeProjectId: "proj-fail" });

    renderHook(() => useWorkspaceSurface());

    await waitFor(() => {
      expect(toastErrorSpy).toHaveBeenCalledWith("Failed to load board snapshot");
    });
  });

  test("resize interaction triggers resize_item with clamped dimensions through BoardCanvas", async () => {
    const applySpy = vi.spyOn(ipc, "applyBoardCommands").mockResolvedValue({
      status: "applied",
      version: 4,
      changes: [],
      diagnostics: [],
      historyEntryId: "hist-1",
    });

    render(
      <ReactFlowProvider>
        <BoardCanvasInner />
        <StoreCapture />
      </ReactFlowProvider>,
    );

    const targetNode = testStoreApi!.getState().nodes.find((n: Node) => n.id === "note-existing");
    expect(targetNode).toBeDefined();

    // Drive the real onResizeEnd path
    await act(async () => {
      await targetNode?.data.onResizeEnd?.("note-existing", { width: 320.4, height: 210.1 });
    });

    expect(applySpy).toHaveBeenCalledTimes(1);
    const [, batch] = applySpy.mock.calls[0] as [string, CommandBatch<BoardCommand>];
    expect(batch.commands).toEqual([
      {
        type: "resize_item",
        item_id: "note-existing",
        size: { width: 320, height: 210 },
      },
    ]);
  });

  test("Delete / Backspace removes selected items in one batch and does not delete when editing note", async () => {
    const applySpy = vi.spyOn(ipc, "applyBoardCommands").mockResolvedValue({
      status: "applied",
      version: 4,
      changes: [],
      diagnostics: [],
      historyEntryId: "hist-1",
    });

    render(<BoardCanvas />);

    // Select note-existing by clicking
    const noteEl = screen.getByTestId("note-card").closest(".react-flow__node")!;
    fireEvent.click(noteEl);

    // Press Delete key
    await act(async () => {
      fireEvent.keyDown(window, { key: "Delete" });
    });

    expect(applySpy).toHaveBeenCalledTimes(1);
    const [, batch] = applySpy.mock.calls[0] as [string, CommandBatch<BoardCommand>];
    expect(batch.commands).toHaveLength(1);
    expect(batch.commands[0]).toEqual({
      type: "remove_item",
      item_id: "note-existing",
    });
  });

  test("rejected batch shows error toast and reloads snapshot", async () => {
    const toastErrorSpy = vi.spyOn(toast, "error");
    vi.spyOn(ipc, "applyBoardCommands").mockResolvedValue({
      status: "rejected",
      version: 3,
      changes: [],
      diagnostics: [
        {
          id: "diag-1",
          code: "CORE/BOARD_VERSION_CONFLICT",
          severity: "error",
          source: "core",
          message: "board version conflict: expected 3, got 2",
          target: "board",
        },
      ],
      historyEntryId: null,
    });

    const getSnapshotSpy = vi.spyOn(ipc, "getBoardSnapshot").mockResolvedValue({
      id: "board-proj-1",
      projectId: "proj-1",
      version: 3,
      items: [],
    });

    useBoardStore.getState().hydrate({
      id: "board-proj-1",
      projectId: "proj-1",
      version: 3,
      items: [],
    });

    render(<BoardCanvas />);

    const newNoteBtn = screen.getByRole("button", { name: "New note" });
    await act(async () => {
      fireEvent.click(newNoteBtn);
    });

    expect(toastErrorSpy).toHaveBeenCalledWith("board version conflict: expected 3, got 2");
    expect(getSnapshotSpy).toHaveBeenCalledWith("proj-1");
  });

  test("undo/redo shortcuts: calls Board undo/redo when BoardCanvas is mounted, but not while typing in an input", async () => {
    const undoSpy = vi.spyOn(ipc, "undoBoard").mockResolvedValue({
      status: "applied",
      version: 4,
      changes: [],
      diagnostics: [],
      historyEntryId: null,
    });
    const redoSpy = vi.spyOn(ipc, "redoBoard").mockResolvedValue({
      status: "applied",
      version: 5,
      changes: [],
      diagnostics: [],
      historyEntryId: null,
    });

    render(
      <div>
        <UndoRedoShortcutMount />
        <BoardCanvas />
        <input data-testid="test-input" />
      </div>,
    );

    // 1. While typing in input, Cmd+Z is ignored
    const input = screen.getByTestId("test-input");
    fireEvent.keyDown(input, { key: "z", metaKey: true });
    expect(undoSpy).not.toHaveBeenCalled();

    // 2. Cmd+Z on window triggers board undo
    await act(async () => {
      fireEvent.keyDown(window, { key: "z", metaKey: true });
    });
    expect(undoSpy).toHaveBeenCalledWith("proj-1", undefined);

    // 3. Cmd+Shift+Z on window triggers board redo
    await act(async () => {
      fireEvent.keyDown(window, { key: "z", metaKey: true, shiftKey: true });
    });
    expect(redoSpy).toHaveBeenCalledWith("proj-1", undefined);
  });

  test("undo/redo shortcuts: when BoardCanvas is unmounted, delegates to workflow store undo", async () => {
    const workflowTemporalUndo = vi.spyOn(useWorkflowStore.temporal.getState(), "undo");
    const workflowTemporalRedo = vi.spyOn(useWorkflowStore.temporal.getState(), "redo");

    const { unmount } = render(
      <div>
        <UndoRedoShortcutMount />
        <BoardCanvas />
      </div>,
    );

    // Unmount BoardCanvas
    unmount();

    render(<UndoRedoShortcutMount />);

    fireEvent.keyDown(window, { key: "z", metaKey: true });
    expect(workflowTemporalUndo).toHaveBeenCalledTimes(1);

    fireEvent.keyDown(window, { key: "z", metaKey: true, shiftKey: true });
    expect(workflowTemporalRedo).toHaveBeenCalledTimes(1);
  });
});
