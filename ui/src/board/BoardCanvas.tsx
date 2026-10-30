import { useCallback, useEffect, useRef, useState } from "react";
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  ReactFlowProvider,
  useReactFlow,
  useNodesState,
  type Node,
  type NodeTypes,
  type OnNodeDrag,
  type Viewport,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { toast } from "sonner";
import * as m from "$paraglide/messages";
import { cn } from "@/lib/utils";
import { useProjectStore } from "@/store/project";
import { useBoardStore } from "@/store/board";
import { useUIStore } from "@/store/uiStore";
import { BoardNode, type BoardNodeData } from "./BoardNode";
import { CanvasEmptyState } from "./CanvasEmptyState";
import { CanvasHud, type CanvasTool } from "./CanvasHud";
import {
  animateViewport,
  CANVAS_FOCUS_MS,
  CANVAS_MAX_ZOOM,
  CANVAS_MIN_ZOOM,
  SESSION_PANEL_COVERAGE,
} from "./canvasView";
import {
  buildAddItemCommand,
  buildDragStopCommands,
  buildRemoveItemCommand,
  buildResizeEndCommand,
  buildUpdateNoteCommand,
  createBoardBatch,
  DEFAULT_NOTE_SIZE,
  roundPosition,
  type BoardCommand,
} from "./batch";
import type { BoardItem } from "@/ipc/schemas";

const nodeTypes: NodeTypes = {
  boardNode: BoardNode,
};

/** How long after placing a note a following double-click is treated as part of it. */
const NOTE_CLICK_GUARD_MS = 500;

/** Viewport a project starts at before it has been panned or zoomed. */
const DEFAULT_VIEWPORT: Viewport = { x: 0, y: 0, zoom: 1 };

const TOOL_CURSOR_CLASS: Record<CanvasTool, string> = {
  select: "canvas-tool-select",
  hand: "canvas-tool-hand",
  note: "canvas-tool-note",
};

type PendingMove = {
  x: number;
  y: number;
  batchId: string;
  resultVersion?: number;
};

export function BoardCanvasInner() {
  const board = useBoardStore();
  const { screenToFlowPosition, getViewport, getZoom, setViewport } = useReactFlow();

  const [editingId, setEditingId] = useState<string | null>(null);
  const [tool, setTool] = useState<CanvasTool>("select");
  const flowWrapperRef = useRef<HTMLDivElement | null>(null);
  const lastNoteAtRef = useRef(0);
  const focusAnimationRef = useRef<(() => void) | null>(null);

  const sessionPanelOpen = useUIStore((state) => state.sessionPanelOpen);
  const canvasFocusItemId = useUIStore((state) => state.canvasFocusItemId);
  const clearCanvasFocus = useUIStore((state) => state.clearCanvasFocus);
  const setCanvasViewport = useUIStore((state) => state.setCanvasViewport);
  const projectId = useProjectStore((state) => state.activeProjectId);

  const pendingBatchRef = useRef<Promise<unknown>>(Promise.resolve());
  const pendingMovesRef = useRef<Map<string, PendingMove>>(new Map());

  // React Flow reads `defaultViewport` once per mount; a canvas that remounts
  // after visiting the library starts at the view this project was left at.
  const initialViewportRef = useRef<Viewport>(
    useUIStore.getState().canvasViewports[projectId] ?? DEFAULT_VIEWPORT,
  );
  const previousProjectRef = useRef(projectId);

  // Switching workspace while the canvas is mounted restores that project's
  // viewport; the first pass is left to `defaultViewport` so a focus request
  // arriving with the mount is not overwritten.
  useEffect(() => {
    if (previousProjectRef.current === projectId) return;
    previousProjectRef.current = projectId;
    const stored = useUIStore.getState().canvasViewports[projectId] ?? DEFAULT_VIEWPORT;
    void setViewport(stored, { duration: 0 });
  }, [projectId, setViewport]);

  // Drops the optimistic positions recorded for one drag batch, or all of
  // them when the failed batch was not a drag.
  const dropPendingMoves = useCallback((batchId: string | undefined) => {
    if (!batchId) {
      pendingMovesRef.current.clear();
      return;
    }
    for (const [id, entry] of pendingMovesRef.current.entries()) {
      if (entry.batchId === batchId) pendingMovesRef.current.delete(id);
    }
  }, []);

  // Batches run one at a time so each reads the version left by the previous one.
  const executeBatch = useCallback(
    (commands: BoardCommand[], batchId?: string) => {
      const run = async () => {
        const batch = createBoardBatch(useBoardStore.getState().version, commands);
        const targetProjectId = useProjectStore.getState().activeProjectId;
        const reloadIfStillActive = async () => {
          if (useProjectStore.getState().activeProjectId !== targetProjectId) return;
          try {
            await useBoardStore.getState().load(targetProjectId);
          } catch {
            toast.error(m["board.loadFailed"]());
          }
        };
        try {
          const result = await useBoardStore.getState().apply(targetProjectId, batch);
          if (result.status === "rejected") {
            dropPendingMoves(batchId);
            const firstDiag = result.diagnostics?.[0] as { message?: string } | undefined;
            toast.error(firstDiag?.message ?? m["board.commandRejected"]());
            await reloadIfStillActive();
          } else if (result.status === "no_op") {
            if (batchId) dropPendingMoves(batchId);
          } else if (batchId) {
            // The snapshot carrying this batch may already have been applied.
            if (useBoardStore.getState().snapshotVersion >= result.version) {
              dropPendingMoves(batchId);
            } else {
              for (const entry of pendingMovesRef.current.values()) {
                if (entry.batchId === batchId) entry.resultVersion = result.version;
              }
            }
          }
        } catch (err: unknown) {
          dropPendingMoves(batchId);
          toast.error(err instanceof Error ? err.message : String(err));
          await reloadIfStillActive();
        }
      };

      const nextPromise = pendingBatchRef.current.then(run, run);
      pendingBatchRef.current = nextPromise;
      return nextPromise;
    },
    [dropPendingMoves],
  );

  const handleStartEdit = useCallback((id: string) => setEditingId(id), []);
  const handleEndEdit = useCallback(() => setEditingId(null), []);

  const handleUpdateNote = useCallback(
    async (id: string, content: string) => {
      const item = useBoardStore.getState().items.find((i) => i.id === id);
      if (!item || item.kind.type !== "note") return;
      if (item.kind.content === content) return;
      const cmd = buildUpdateNoteCommand(id, content);
      await executeBatch([cmd]);
    },
    [executeBatch],
  );

  const handleResizeEnd = useCallback(
    async (id: string, size: { width: number; height: number }) => {
      const item = useBoardStore.getState().items.find((i) => i.id === id);
      const cmd = buildResizeEndCommand(item, size);
      if (!cmd) return;
      await executeBatch([cmd]);
    },
    [executeBatch],
  );

  const createOrReconcileNode = useCallback(
    (item: BoardItem, prev?: Node, isEditing?: boolean, handActive?: boolean): Node => {
      let position = { x: item.position.x, y: item.position.y };
      const optimistic = pendingMovesRef.current.get(item.id);
      if (optimistic) {
        position = { x: optimistic.x, y: optimistic.y };
      } else if (prev?.dragging) {
        position = prev.position;
      }

      return {
        id: item.id,
        type: "boardNode",
        position,
        selected: prev ? Boolean(prev.selected) : false,
        dragging: prev ? Boolean(prev.dragging) : false,
        data: {
          item,
          isEditing: isEditing ?? false,
          onStartEdit: handleStartEdit,
          onEndEdit: handleEndEdit,
          onUpdateNote: (id: string, content: string) => void handleUpdateNote(id, content),
          onResizeEnd: (id: string, size: { width: number; height: number }) =>
            void handleResizeEnd(id, size),
        } as BoardNodeData,
        style: {
          width: item.size.width,
          height: item.size.height,
          zIndex: item.z,
        },
        // Per-node flags win over the flow-wide props, so the hand tool has
        // to be reflected here as well as in nodesDraggable/elementsSelectable.
        draggable: !item.locked && !isEditing && !handActive,
        selectable: !isEditing && !handActive,
      };
    },
    [handleEndEdit, handleResizeEnd, handleStartEdit, handleUpdateNote],
  );

  const [nodes, setNodes, onNodesChange] = useNodesState<Node>(
    (() => {
      const items = useBoardStore.getState().items;
      const sorted = [...items].sort((a, b) => a.z - b.z);
      return sorted.map((item) => createOrReconcileNode(item, undefined, false));
    })(),
  );

  // Reconcile local nodes when board.items, board.version, snapshotVersion, or editingId changes
  const prevBoardKeyRef = useRef(`${board.projectId}:${board.id}`);
  useEffect(() => {
    const key = `${board.projectId}:${board.id}`;
    if (prevBoardKeyRef.current !== key) {
      prevBoardKeyRef.current = key;
      pendingMovesRef.current.clear();
    }

    const currentSnapshotVersion = board.snapshotVersion;
    for (const [id, entry] of pendingMovesRef.current.entries()) {
      if (entry.resultVersion !== undefined && currentSnapshotVersion >= entry.resultVersion) {
        pendingMovesRef.current.delete(id);
      }
    }

    setNodes((prevNodes) => {
      const prevMap = new Map(prevNodes.map((n) => [n.id, n]));
      const sorted = [...board.items].sort((a, b) => a.z - b.z);
      return sorted.map((item) => {
        const prev = prevMap.get(item.id);
        const isEditing = editingId === item.id;
        return createOrReconcileNode(item, prev, isEditing, tool === "hand");
      });
    });
  }, [
    board.projectId,
    board.id,
    board.items,
    board.version,
    board.snapshotVersion,
    editingId,
    tool,
    createOrReconcileNode,
    setNodes,
  ]);

  const handleNodeDragStop: OnNodeDrag = useCallback(
    async (_event, node, draggedNodes) => {
      const targetNodes = draggedNodes && draggedNodes.length > 0 ? draggedNodes : [node];
      const batchId = crypto.randomUUID();
      for (const n of targetNodes) {
        const rounded = roundPosition(n.position);
        pendingMovesRef.current.set(n.id, {
          x: rounded.x,
          y: rounded.y,
          batchId,
        });
      }
      setNodes((prevNodes) =>
        prevNodes.map((n) => {
          const move = targetNodes.find((m) => m.id === n.id);
          return move ? { ...n, position: roundPosition(move.position) } : n;
        }),
      );
      const commands = buildDragStopCommands(useBoardStore.getState().items, targetNodes);
      if (commands.length === 0) {
        dropPendingMoves(batchId);
        return;
      }
      await executeBatch(commands, batchId);
    },
    [dropPendingMoves, executeBatch, setNodes],
  );

  // Unmount cleanup: a focus animation must not keep writing into a disposed flow.
  useEffect(() => () => focusAnimationRef.current?.(), []);

  // A navigation back from the library remounts this board with the id already
  // requested, so the effect also runs on mount. It waits for the snapshot
  // before treating an unknown id as stale.
  useEffect(() => {
    if (!canvasFocusItemId || board.id === "") return;
    const item = board.items.find((candidate) => candidate.id === canvasFocusItemId);
    if (!item) {
      clearCanvasFocus();
      return;
    }
    const wrapper = flowWrapperRef.current;
    if (!wrapper) {
      clearCanvasFocus();
      return;
    }
    const width = wrapper.clientWidth;
    const height = wrapper.clientHeight;
    if (width <= 1 || height <= 1) {
      clearCanvasFocus();
      return;
    }
    const coverage = sessionPanelOpen ? SESSION_PANEL_COVERAGE : 0;
    const currentZoom = getZoom();
    const targetZoom = currentZoom < 0.5 ? 1 : currentZoom;
    const target = {
      x: (width - coverage) / 2 - (item.position.x + item.size.width / 2) * targetZoom,
      y: height / 2 - (item.position.y + item.size.height / 2) * targetZoom,
      zoom: targetZoom,
    };
    focusAnimationRef.current?.();
    focusAnimationRef.current = animateViewport(
      getViewport(),
      target,
      setViewport,
      CANVAS_FOCUS_MS,
    );
    clearCanvasFocus();
  }, [
    board.id,
    board.items,
    canvasFocusItemId,
    clearCanvasFocus,
    getViewport,
    getZoom,
    sessionPanelOpen,
    setViewport,
  ]);

  const createNewNote = useCallback(
    async (point: { x: number; y: number }) => {
      lastNoteAtRef.current = Date.now();
      const newItemId = crypto.randomUUID();
      const cmd = buildAddItemCommand({
        itemId: newItemId,
        kind: { type: "note", content: "" },
        position: {
          x: point.x - DEFAULT_NOTE_SIZE.width / 2,
          y: point.y - DEFAULT_NOTE_SIZE.height / 2,
        },
        size: DEFAULT_NOTE_SIZE,
      });
      setEditingId(newItemId);
      await executeBatch([cmd]);
    },
    [executeBatch],
  );

  const createNoteAtVisibleCenter = useCallback(() => {
    const wrapper = flowWrapperRef.current;
    if (!wrapper) return;
    const rect = wrapper.getBoundingClientRect();
    const coverage = sessionPanelOpen ? SESSION_PANEL_COVERAGE : 0;
    const point = screenToFlowPosition({
      x: rect.left + (rect.width - coverage) / 2,
      y: rect.top + rect.height / 2,
    });
    void createNewNote(point);
  }, [createNewNote, screenToFlowPosition, sessionPanelOpen]);

  const handlePaneClick = useCallback(
    (event: React.MouseEvent) => {
      if (tool !== "note") return;
      const point = screenToFlowPosition({ x: event.clientX, y: event.clientY });
      void createNewNote(point);
      setTool("select");
    },
    [createNewNote, screenToFlowPosition, tool],
  );

  const handleDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      // A double-click that follows a note-tool click would otherwise place a
      // second note on the same spot.
      if (Date.now() - lastNoteAtRef.current < NOTE_CLICK_GUARD_MS) return;
      const target = e.target as HTMLElement | null;
      if (
        target?.closest(
          ".react-flow__node,button,a,input,textarea,select,[role='toolbar']," +
            "[data-testid='canvas-empty-state'],[data-testid='canvas-dock'],[data-testid='canvas-zoom']",
        )
      ) {
        return;
      }
      const flowPos = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      void createNewNote(flowPos);
    },
    [screenToFlowPosition, createNewNote],
  );

  // Keyboard shortcut: Delete / Backspace removes selected items
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (editingId) return;
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)
      ) {
        return;
      }
      if (e.key === "Delete" || e.key === "Backspace") {
        const selected = nodes.filter((n) => n.selected);
        if (selected.length === 0) return;
        e.preventDefault();
        const commands = selected.map((n) => buildRemoveItemCommand(n.id));
        void executeBatch(commands);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [editingId, nodes, executeBatch]);

  return (
    <div
      ref={flowWrapperRef}
      className={cn("relative h-full w-full bg-ground", TOOL_CURSOR_CLASS[tool])}
      onDoubleClick={handleDoubleClick}
      data-testid="board-canvas"
    >
      <ReactFlow
        nodes={nodes}
        edges={[]}
        onNodesChange={onNodesChange}
        onNodeDragStop={handleNodeDragStop}
        onMoveEnd={(_, viewport) => setCanvasViewport(projectId, viewport)}
        onPaneClick={handlePaneClick}
        defaultViewport={initialViewportRef.current}
        nodeTypes={nodeTypes}
        selectionOnDrag={tool === "select"}
        panOnDrag={tool === "hand" ? true : [1, 2]}
        nodesDraggable={tool !== "hand"}
        elementsSelectable={tool !== "hand"}
        zoomOnDoubleClick={false}
        deleteKeyCode={null}
        minZoom={CANVAS_MIN_ZOOM}
        maxZoom={CANVAS_MAX_ZOOM}
        fitView={false}
        className="h-full w-full"
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1}
          color="var(--color-grid)"
          bgColor="var(--color-ground)"
        />
      </ReactFlow>

      {board.id !== "" && board.items.length === 0 && (
        <CanvasEmptyState onCreateNote={createNoteAtVisibleCenter} />
      )}
      <CanvasHud tool={tool} onToolChange={setTool} />
    </div>
  );
}

export function BoardCanvas() {
  return (
    <ReactFlowProvider>
      <BoardCanvasInner />
    </ReactFlowProvider>
  );
}
