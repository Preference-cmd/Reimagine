import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  type NodeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { useCallback } from "react";
import { ChevronDown } from "lucide-react";
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
   input that mirrors the reference. No visible sockets; the connection
   to the Image Generator is via a hidden handle. */
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
          <div className="mb-2.5 text-[12px] leading-relaxed text-zinc-300">
            {d.prompt}
          </div>
          <input
            className="w-full rounded-md border border-white/5 bg-zinc-900/60 px-2.5 py-1.5 text-[11px] text-zinc-400 placeholder-zinc-600 focus:outline-none focus:ring-1 focus:ring-white/10"
            placeholder="Type what you want to get"
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
        <span className="flex items-center gap-1.5 rounded-md bg-zinc-800/70 px-2.5 py-1.5 text-[11px] font-medium leading-none text-zinc-100">
          <span className="truncate">
            {d.parameters?.[0]?.value ?? ""}
          </span>
          <ChevronDown className="h-3 w-3 shrink-0 text-zinc-500" />
        </span>
      </div>
    </BaseNode>
  );
};

/* Image Generator node — three inputs on the left, one image output on
   the right, and a stack of control rows in the inner card. */
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
          className="aspect-square w-full rounded-md border border-white/5 object-cover"
          src="https://images.unsplash.com/photo-1530595467537-0b5996c41f2d?w=320&q=70"
          alt="generated"
        />
        <div className="absolute inset-x-2 bottom-2 truncate rounded bg-black/60 px-2 py-1 text-[10px] text-zinc-400 backdrop-blur-md">
          Final Result
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

/* ───── Component ───── */

import type { NodeProps } from "@xyflow/react";

export function NodeCanvas() {
  const nodes = useWorkflowStore((s) => s.nodes);
  const edges = useWorkflowStore((s) => s.edges);
  const onNodesChange = useWorkflowStore((s) => s.onNodesChange);
  const onEdgesChange = useWorkflowStore((s) => s.onEdgesChange);
  const onConnect = useWorkflowStore((s) => s.onConnect);

  const handleSelection = useCallback(
    ({ nodes: selected }: { nodes: Array<{ id: string; type?: string | null }> }) => {
      const node = selected[0];
      onNodeSelect(node ? { id: node.id, type: node.type ?? null } : null);
    },
    [],
  );

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
        fitView
        proOptions={{ hideAttribution: true }}
        minZoom={0.3}
        maxZoom={2}
        colorMode="dark"
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1.5}
          color="#2a2a2a"
        />
        <Controls
          position="bottom-left"
          showInteractive={false}
          className="!bg-zinc-900/80 !border !border-white/5 !shadow-xl [&>button]:!bg-transparent [&>button]:!border-white/5 [&>button]:!text-zinc-400 [&>button:hover]:!bg-zinc-800 [&>button:hover]:!text-white"
        />
        <MiniMap
          position="bottom-right"
          pannable
          zoomable
          maskColor="rgba(0,0,0,0.7)"
          className="!bg-zinc-900/80 !border !border-white/5 !rounded-lg"
        />
      </ReactFlow>
    </div>
  );
}