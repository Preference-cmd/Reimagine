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
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Plus } from "lucide-react";
import { toast } from "sonner";
import * as m from "$paraglide/messages";
import { useProjectStore } from "@/store/project";
import { useBoardStore } from "@/store/board";
import { registerBoardUndoTarget, unregisterBoardUndoTarget } from "@/hooks/useUndoRedo";
import { BoardNode, type BoardNodeData } from "./BoardNode";
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

type PendingMove = {
  x: number;
  y: number;
  batchId: string;
  resultVersion?: number;
};

export function BoardCanvasInner() {
  const board = useBoardStore();
  const { screenToFlowPosition } = useReactFlow();

  const [editingId, setEditingId] = useState<string | null>(null);
  const flowWrapperRef = useRef<HTMLDivElement | null>(null);

  const pendingBatchRef = useRef<Promise<unknown>>(Promise.resolve());
  const pendingMovesRef = useRef<Map<string, PendingMove>>(new Map());

  // Register undo/redo shortcuts target on mount and unregister on unmount
  useEffect(() => {
    const target = {
      undo: () => useBoardStore.getState().undo(useProjectStore.getState().activeProjectId),
      redo: () => useBoardStore.getState().redo(useProjectStore.getState().activeProjectId),
    };
    registerBoardUndoTarget(target);
    return () => unregisterBoardUndoTarget(target);
  }, []);

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
    (item: BoardItem, prev?: Node, isEditing?: boolean): Node => {
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
        draggable: !item.locked && !isEditing,
        selectable: !isEditing,
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
        return createOrReconcileNode(item, prev, isEditing);
      });
    });
  }, [
    board.projectId,
    board.id,
    board.items,
    board.version,
    board.snapshotVersion,
    editingId,
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

  const createNewNote = useCallback(
    async (pos: { x: number; y: number }) => {
      const newItemId = crypto.randomUUID();
      const cmd = buildAddItemCommand({
        itemId: newItemId,
        kind: { type: "note", content: "" },
        position: pos,
        size: DEFAULT_NOTE_SIZE,
      });
      setEditingId(newItemId);
      await executeBatch([cmd]);
    },
    [executeBatch],
  );

  const handleNewNoteClick = useCallback(() => {
    const center = flowWrapperRef.current
      ? {
          x: flowWrapperRef.current.clientWidth / 2,
          y: flowWrapperRef.current.clientHeight / 2,
        }
      : { x: window.innerWidth / 2, y: window.innerHeight / 2 };
    const flowPos = screenToFlowPosition(center);
    void createNewNote(flowPos);
  }, [screenToFlowPosition, createNewNote]);

  const handleDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (target?.closest(".react-flow__node")) return;
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
      className="relative h-full w-full bg-background"
      onDoubleClick={handleDoubleClick}
      data-testid="board-canvas"
    >
      {/* Floating Toolbar */}
      <div className="absolute top-4 left-4 z-20 flex items-center gap-2 rounded-xl border border-outline bg-surface/85 px-3 py-1.5 shadow-xs backdrop-blur-md">
        <button
          type="button"
          onClick={handleNewNoteClick}
          className="flex h-7 cursor-pointer items-center gap-1.5 rounded-lg bg-primary px-2.5 text-caption font-medium text-primary-foreground transition-opacity hover:opacity-90"
          data-testid="toolbar-new-note"
        >
          <Plus className="h-3.5 w-3.5" />
          <span>{m["board.newNote"]()}</span>
        </button>
      </div>

      <ReactFlow
        nodes={nodes}
        edges={[]}
        onNodesChange={onNodesChange}
        onNodeDragStop={handleNodeDragStop}
        nodeTypes={nodeTypes}
        zoomOnDoubleClick={false}
        deleteKeyCode={null}
        minZoom={0.2}
        maxZoom={2.5}
        fitView={false}
        className="h-full w-full"
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} />
      </ReactFlow>
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
