import {
  ReactFlow,
  Background,
  BackgroundVariant,
  MiniMap,
  ReactFlowProvider,
  useReactFlow,
  type FitViewOptions,
  type NodeProps,
  type NodeTypes,
  type Viewport,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { useCallback, useEffect, useMemo, useState } from "react";
import { ChevronDown, Maximize2, Minus, Plus } from "lucide-react";
import { toast } from "sonner";
import { BaseNode, type ParamRow, type SocketSlot } from "./BaseNode";
import { edgeTypes } from "./FlowEdge";
import { GenericNode } from "./GenericNode";
import { NodePreview } from "./NodePreview";
import { useWorkflowStore, onNodeSelect } from "@/store/workflow";
import { useNodeRegistryStore } from "@/store/nodeRegistry";
import { useNodeArtifact } from "@/hooks/useNodeArtifact";
import { checkConnection } from "@/lib/socketCompat";
import { createNodeAt, NODE_DRAG_MIME } from "@/lib/nodeFactory";
import { flowPositionFor, registerFlowInstance, selectAllNodes } from "@/lib/flowInstance";
import { useUIStore } from "@/store/uiStore";

/* ───── Demo node data ───── */

type DemoNodeData = {
  title: string;
  tone: string;
  inputs?: SocketSlot[];
  outputs?: SocketSlot[];
  prompt?: string;
  parameters?: ParamRow[];
  disabled?: unknown;
};

/* Prompt node — Positive / Negative. The body is free text + a faux
   prompt field. No visible sockets; the connection to the sampler is
   via a hidden handle. */
const PromptNode = ({ data, selected }: NodeProps) => {
  const d = data as unknown as DemoNodeData;
  return (
    <BaseNode title={d.title} tone={d.tone} selected={selected} disabled={d.disabled === true}>
      {d.prompt && (
        <>
          <div className="mb-2.5 text-body-sm leading-relaxed text-on-surface">{d.prompt}</div>
          <input
            className="w-full rounded-md border border-control-border bg-surface-container-high px-2.5 py-1.5 text-body-sm text-on-surface placeholder-on-surface-variant focus:outline-none focus:ring-1 focus:ring-control-active"
            placeholder="Edit prompt"
            readOnly
          />
        </>
      )}
    </BaseNode>
  );
};

/* Model node — three outputs (model / positive / negative) on the right
   and a single model-selector dropdown in the inner card, right-aligned. */
const ModelNode = ({ data, selected }: NodeProps) => {
  const d = data as unknown as DemoNodeData;
  return (
    <BaseNode
      title={d.title}
      tone={d.tone}
      outputs={d.outputs}
      selected={selected}
      disabled={d.disabled === true}
    >
      <div className="flex items-center justify-end">
        <span className="flex items-center gap-1.5 rounded-md bg-surface-container-high px-2.5 py-1.5 text-body-sm font-medium leading-none text-on-surface">
          <span className="truncate">{d.parameters?.[0]?.value ?? ""}</span>
          <ChevronDown className="h-3 w-3 shrink-0 text-on-surface-variant" />
        </span>
      </div>
    </BaseNode>
  );
};

/* Sampler node — conditioning and latent inputs on the left, one image
   output on the right, and a compact stack of sampling parameters. */
const ImageGeneratorNode = ({ data, selected }: NodeProps) => {
  const d = data as unknown as DemoNodeData;
  return (
    <BaseNode
      title={d.title}
      tone={d.tone}
      inputs={d.inputs}
      outputs={d.outputs}
      parameters={d.parameters}
      selected={selected}
      disabled={d.disabled === true}
    />
  );
};

/* Image output node — single image input and a preview in the inner card.
   Shows the latest run artifact inline (F5-4) once a run produces one. */
const ImageOutputNode = ({ data, selected, id }: NodeProps) => {
  const d = data as unknown as DemoNodeData;
  const preview = useNodeArtifact(id);
  return (
    <BaseNode
      title={d.title}
      tone={d.tone}
      inputs={d.inputs}
      selected={selected}
      disabled={d.disabled === true}
    >
      <div className="relative">
        <div className="aspect-square w-full overflow-hidden rounded-md border border-control-border">
          {preview?.status === "ready" && preview.url ? (
            <img
              className="h-full w-full object-cover"
              src={preview.url}
              alt={preview.artifactId}
            />
          ) : (
            <NodePreview preview={preview} />
          )}
        </div>
        <div className="absolute inset-x-2 bottom-2 truncate rounded bg-preview-scrim px-2 py-1 text-caption text-white backdrop-blur-md">
          Preview
        </div>
      </div>
    </BaseNode>
  );
};

/* ───── NodeTypes registry (F2-1) ─────
   Hand-crafted components stay for the types they cover; every catalog
   type without a dedicated component maps to GenericNode, and unknown
   types (catalog fetch failure, stale persisted graphs) fall through to
   GenericNode via the `default` slot so the canvas never crashes. */

const handCraftedNodeTypes: NodeTypes = {
  prompt: PromptNode,
  model: ModelNode,
  imageGenerator: ImageGeneratorNode,
  imageOutput: ImageOutputNode,
};

const canvasFitViewOptions = { padding: 0.22 } satisfies FitViewOptions;

/* ───── Zoom controls (horizontal: − 100% + ⊡) ───── */

function ZoomControls({ zoom }: { zoom: number }) {
  const { zoomIn, zoomOut, zoomTo, fitView } = useReactFlow();
  const iconButton =
    "flex h-8 w-8 cursor-pointer items-center justify-center rounded-full text-on-surface-variant transition-colors hover:bg-control-hover hover:text-on-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30";

  return (
    <div className="panel-flat absolute bottom-5 left-5 z-10 flex h-10 items-center gap-0.5 rounded-xl px-1 shadow-[0_2px_8px_-2px_rgb(0_0_0/0.15)]">
      <button
        type="button"
        onClick={() => zoomOut({ duration: 200 })}
        aria-label="Zoom out"
        className={iconButton}
      >
        <Minus className="h-3.5 w-3.5" />
      </button>
      <button
        type="button"
        onClick={() => zoomTo(1, { duration: 200 })}
        aria-label="Reset zoom to 100%"
        className="flex h-8 min-w-12 cursor-pointer items-center justify-center rounded-full px-2 text-body-sm font-medium tabular-nums text-on-surface transition-colors hover:bg-control-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30"
      >
        {Math.round(zoom * 100)}%
      </button>
      <button
        type="button"
        onClick={() => zoomIn({ duration: 200 })}
        aria-label="Zoom in"
        className={iconButton}
      >
        <Plus className="h-3.5 w-3.5" />
      </button>
      <button
        type="button"
        onClick={() => fitView({ duration: 200, padding: 0.22 })}
        aria-label="Fit view"
        className={iconButton}
      >
        <Maximize2 className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}

/* ───── Canvas flow (inside ReactFlowProvider) ───── */

function CanvasFlow({ themeMode }: { themeMode: "light" | "dark" }) {
  const nodes = useWorkflowStore((s) => s.nodes);
  const edges = useWorkflowStore((s) => s.edges);
  const onNodesChange = useWorkflowStore((s) => s.onNodesChange);
  const onEdgesChange = useWorkflowStore((s) => s.onEdgesChange);
  const onConnect = useWorkflowStore((s) => s.onConnect);
  const removeNodes = useWorkflowStore((s) => s.removeNodes);
  const duplicateNode = useWorkflowStore((s) => s.duplicateNode);
  const toggleNodeDisabled = useWorkflowStore((s) => s.toggleNodeDisabled);
  const disconnectNodeEdges = useWorkflowStore((s) => s.disconnectNodeEdges);
  const removeEdges = useWorkflowStore((s) => s.removeEdges);
  const catalogDefs = useNodeRegistryStore((s) => s.defs);
  const openContextMenu = useUIStore((s) => s.openContextMenu);
  const closeContextMenu = useUIStore((s) => s.closeContextMenu);
  const openNodePalette = useUIStore((s) => s.openNodePalette);
  const startRename = useUIStore((s) => s.startRename);
  const instance = useReactFlow();
  const { fitView, screenToFlowPosition } = instance;
  const [zoom, setZoom] = useState(1);

  /* Register the flow instance for out-of-tree callers (palette, explorer,
     context menus) — F2-2/F3-3. The instance object is stable for the
     provider's lifetime, so an effect-only registration is enough. */
  useEffect(() => {
    registerFlowInstance(instance);
  }, [instance]);
  /* Schema-driven registry: hand-crafted first, catalog types via
     GenericNode, unknown types via the `default` slot (F2-1). */
  const nodeTypes = useMemo(() => {
    const registry: NodeTypes = { ...handCraftedNodeTypes };
    for (const def of catalogDefs.values()) {
      if (!registry[def.type]) registry[def.type] = GenericNode;
    }
    registry.default = GenericNode;
    return registry;
  }, [catalogDefs]);

  /* Live connection validation (F2-3) — feeds React Flow's validity
     highlighting on handles during a drag, and the onConnect gate. */
  const isValidConnection = useCallback(
    (connection: Parameters<typeof checkConnection>[0]): boolean =>
      checkConnection(connection, nodes, catalogDefs).ok,
    [catalogDefs, nodes],
  );

  const handleConnect = useCallback(
    (conn: Parameters<typeof onConnect>[0]) => {
      const result = checkConnection(conn, nodes, catalogDefs);
      if (!result.ok) {
        const label = (kind: string | null) => (kind ? `"${kind}"` : "unknown");
        toast.error("Connection type mismatch", {
          description: `Cannot connect ${label(result.sourceKind)} to ${label(result.targetKind)}.`,
        });
        return;
      }
      onConnect(conn, {
        sourceKind: result.sourceKind ?? "latent",
        targetKind: result.targetKind ?? "latent",
      });
    },
    [catalogDefs, nodes, onConnect],
  );

  /* Rejected connections never reach `onConnect` (React Flow gates them
     via `isValidConnection`), so surface the rejection here (F2-3). */
  const handleConnectEnd = useCallback(
    (
      _event: MouseEvent | TouchEvent,
      state: {
        isValid?: boolean | null;
        fromNode?: { id: string } | null;
        fromHandle?: { id?: string | null } | null;
        toNode?: { id: string } | null;
        toHandle?: { id?: string | null } | null;
      },
    ) => {
      if (state.isValid !== false || !state.fromNode || !state.toNode) return;
      const result = checkConnection(
        {
          source: state.fromNode.id,
          sourceHandle: state.fromHandle?.id ?? null,
          target: state.toNode.id,
          targetHandle: state.toHandle?.id ?? null,
        },
        nodes,
        catalogDefs,
      );
      if (result.ok) return;
      const label = (kind: string | null) => (kind ? `"${kind}"` : "unknown");
      toast.error("Connection type mismatch", {
        description: `Cannot connect ${label(result.sourceKind)} to ${label(result.targetKind)}.`,
      });
    },
    [catalogDefs, nodes],
  );

  /* ── Drag-and-drop node creation (F2-2) ── */

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  }, []);

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const typeId = event.dataTransfer.getData(NODE_DRAG_MIME);
      if (!typeId) return;
      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });
      if (!createNodeAt(typeId, position)) {
        toast.error("Unknown node type", {
          description: `"${typeId}" is not in the node catalog.`,
        });
      }
    },
    [screenToFlowPosition],
  );

  /* ── Double-click canvas → node picker at cursor (F2-2) ── */

  const handleCanvasDoubleClick = useCallback(
    (event: React.MouseEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.closest(".react-flow__node, .react-flow__edge, .react-flow__minimap")) {
        return;
      }
      closeContextMenu();
      openNodePalette(screenToFlowPosition({ x: event.clientX, y: event.clientY }));
    },
    [closeContextMenu, openNodePalette, screenToFlowPosition],
  );

  /* ── Context menus (F3-1) ── */

  const handlePaneContextMenu = useCallback(
    (event: React.MouseEvent | MouseEvent) => {
      event.preventDefault();
      const x = Math.min(event.clientX, window.innerWidth - 176);
      const y = Math.min(event.clientY, window.innerHeight - 240);
      openContextMenu(x, y, [
        {
          id: "add-node",
          label: "Add Node…",
          shortcut: "Double-click",
          onSelect: () => openNodePalette(flowPositionFor({ x: event.clientX, y: event.clientY })),
        },
        {
          id: "paste",
          label: "Paste",
          disabled: true,
          onSelect: () => {},
        },
        {
          id: "select-all",
          label: "Select All",
          shortcut: "⌘A",
          onSelect: selectAllNodes,
        },
        {
          id: "fit-view",
          label: "Fit View",
          shortcut: "⌘0",
          onSelect: () => void fitView({ duration: 200, padding: 0.22 }),
        },
      ]);
    },
    [fitView, openContextMenu, openNodePalette],
  );

  const handleNodeContextMenu = useCallback(
    (event: React.MouseEvent, node: { id: string; data?: unknown }) => {
      event.preventDefault();
      const data = node.data as { disabled?: unknown; title?: unknown } | undefined;
      const disabled = data?.disabled === true;
      const title = typeof data?.title === "string" && data.title ? data.title : node.id;
      const x = Math.min(event.clientX, window.innerWidth - 176);
      const y = Math.min(event.clientY, window.innerHeight - 240);
      openContextMenu(x, y, [
        {
          id: "duplicate",
          label: "Duplicate",
          shortcut: "⌘D",
          onSelect: () => duplicateNode(node.id),
        },
        {
          id: "toggle-disable",
          label: disabled ? "Enable" : "Disable",
          onSelect: () => toggleNodeDisabled(node.id),
        },
        {
          id: "rename",
          label: "Rename…",
          onSelect: () => startRename({ id: node.id, title }),
        },
        {
          id: "disconnect",
          label: "Disconnect All",
          onSelect: () => disconnectNodeEdges(node.id),
        },
        {
          id: "delete",
          label: "Delete",
          danger: true,
          shortcut: "⌫",
          onSelect: () => removeNodes([node.id]),
        },
      ]);
    },
    [
      disconnectNodeEdges,
      duplicateNode,
      openContextMenu,
      removeNodes,
      startRename,
      toggleNodeDisabled,
    ],
  );

  const handleEdgeContextMenu = useCallback(
    (event: React.MouseEvent, edge: { id: string }) => {
      event.preventDefault();
      const x = Math.min(event.clientX, window.innerWidth - 176);
      const y = Math.min(event.clientY, window.innerHeight - 240);
      openContextMenu(x, y, [
        {
          id: "disconnect",
          label: "Disconnect",
          onSelect: () => removeEdges([edge.id]),
        },
        {
          id: "delete",
          label: "Delete",
          danger: true,
          onSelect: () => removeEdges([edge.id]),
        },
      ]);
    },
    [openContextMenu, removeEdges],
  );

  const handleSelection = useCallback(
    ({ nodes: selected }: { nodes: Array<{ id: string; type?: string | null }> }) => {
      const node = selected[0];
      onNodeSelect(node ? { id: node.id, type: node.type ?? null } : null);
    },
    [],
  );

  const handleMove = useCallback((_: unknown, viewport: Viewport) => {
    setZoom(viewport.zoom);
  }, []);

  return (
    <div className="canvas-grid absolute inset-0 z-0">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        defaultEdgeOptions={{ type: "flow" }}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={handleConnect}
        onConnectEnd={handleConnectEnd}
        isValidConnection={isValidConnection}
        onSelectionChange={handleSelection}
        onMove={handleMove}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onPaneContextMenu={handlePaneContextMenu}
        onNodeContextMenu={handleNodeContextMenu}
        onEdgeContextMenu={handleEdgeContextMenu}
        onNodeDragStart={closeContextMenu}
        onDoubleClick={handleCanvasDoubleClick}
        zoomOnDoubleClick={false}
        fitView
        fitViewOptions={canvasFitViewOptions}
        proOptions={{ hideAttribution: true }}
        minZoom={0.3}
        maxZoom={2}
        colorMode={themeMode}
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1.5}
          color="var(--color-canvas-grid-dot)"
        />
        <ZoomControls zoom={zoom} />
        <MiniMap
          pannable
          zoomable
          bgColor="var(--color-panel-flat)"
          maskColor="transparent"
          nodeColor="var(--color-on-panel-muted)"
          nodeStrokeColor="var(--color-on-panel)"
          className="panel-flat absolute bottom-5 right-5 z-10 !m-0 !h-24 !w-40 !rounded-xl shadow-[0_2px_8px_-2px_rgb(0_0_0/0.15)] p-2.5"
        />
      </ReactFlow>
    </div>
  );
}

/* ───── Component ───── */

export function NodeCanvas({ themeMode }: { themeMode: "light" | "dark" }) {
  return (
    <ReactFlowProvider>
      <CanvasFlow themeMode={themeMode} />
    </ReactFlowProvider>
  );
}
