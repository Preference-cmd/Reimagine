import {
  ReactFlow,
  Background,
  BackgroundVariant,
  MiniMap,
  useReactFlow,
  type FitViewOptions,
  type NodeProps,
  type NodeTypes,
  type Viewport,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { useCallback, useState } from "react";
import { ChevronDown, Maximize2, Minus, Plus } from "lucide-react";
import { BaseNode, type ParamRow, type SocketSlot } from "./BaseNode";
import { edgeTypes } from "./FlowEdge";
import { useWorkflowStore, onNodeSelect } from "@/store/workflow";

/* ───── Demo node data ───── */

type DemoNodeData = {
  title: string;
  tone: string;
  inputs?: SocketSlot[];
  outputs?: SocketSlot[];
  prompt?: string;
  parameters?: ParamRow[];
};

/* Prompt node — Positive / Negative. The body is free text + a faux
   prompt field. No visible sockets; the connection to the sampler is
   via a hidden handle. */
const PromptNode = ({ data, selected }: NodeProps) => {
  const d = data as unknown as DemoNodeData;
  return (
    <BaseNode
      title={d.title}
      tone={d.tone}
      selected={selected}
    >
      {d.prompt && (
        <>
          <div className="mb-2.5 text-body-sm leading-relaxed text-on-surface">
            {d.prompt}
          </div>
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
    >
      <div className="flex items-center justify-end">
        <span className="flex items-center gap-1.5 rounded-md bg-surface-container-high px-2.5 py-1.5 text-body-sm font-medium leading-none text-on-surface">
          <span className="truncate">
            {d.parameters?.[0]?.value ?? ""}
          </span>
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
    />
  );
};

/* Image output node — single image input and a preview in the inner card. */
const ImageOutputNode = ({ data, selected }: NodeProps) => {
  const d = data as unknown as DemoNodeData;
  return (
    <BaseNode
      title={d.title}
      tone={d.tone}
      inputs={d.inputs}
      selected={selected}
    >
      <div className="relative">
        <img
          className="aspect-square w-full rounded-md border border-control-border object-cover"
          src="https://images.unsplash.com/photo-1530595467537-0b5996c41f2d?w=320&q=70"
          alt="generated"
        />
        <div className="absolute inset-x-2 bottom-2 truncate rounded bg-preview-scrim px-2 py-1 text-caption text-white backdrop-blur-md">
          Preview
        </div>
      </div>
    </BaseNode>
  );
};

/* ───── NodeTypes registry ───── */

const nodeTypes: NodeTypes = {
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
    <div className="overlay-slot-canvas-controls panel-flat flex h-11 items-center gap-0.5 rounded-2xl px-1">
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

/* ───── Component ───── */

export function NodeCanvas({ themeMode }: { themeMode: "light" | "dark" }) {
  const nodes = useWorkflowStore((s) => s.nodes);
  const edges = useWorkflowStore((s) => s.edges);
  const onNodesChange = useWorkflowStore((s) => s.onNodesChange);
  const onEdgesChange = useWorkflowStore((s) => s.onEdgesChange);
  const onConnect = useWorkflowStore((s) => s.onConnect);
  const [zoom, setZoom] = useState(1);

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
        onConnect={onConnect}
        onSelectionChange={handleSelection}
        onMove={handleMove}
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
          className="overlay-slot-minimap panel-flat !m-0 !h-24 !w-40 !rounded-2xl p-2.5"
        />
      </ReactFlow>
    </div>
  );
}
