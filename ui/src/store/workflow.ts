import { create } from "zustand";
import { temporal, type TemporalState } from "zundo";
import { useStore } from "zustand";
import type { Node, NodeChange, EdgeChange, Connection } from "@xyflow/react";
import {
  applyNodeChanges,
  applyEdgeChanges,
  addEdge as rfAddEdge,
} from "@xyflow/react";

import type { FlowEdge, FlowEdgeData } from "@/components/canvas/FlowEdge";
import type { ParamValue } from "@/lib/nodes";

export type SelectionInfo = {
  id: string;
  type: string | null;
} | null;

/** Patch of typed parameter values keyed by param id (F2-4). */
export type ParamPatch = Record<string, ParamValue>;

type WorkflowState = {
  /** Stable workflow id — the file name in the workspace `workflows/` dir. */
  id: string;
  name: string;
  nodes: Node[];
  edges: FlowEdge[];
  // ── view state (excluded from undo history) ──────────────────────
  selectedNode: SelectionInfo;
  propertiesPanelOpen: boolean;
  // mutations (all flow through zundo's temporal middleware)
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (conn: Connection) => void;
  onNodeSelect: (s: SelectionInfo) => void;
  setPropertiesPanelOpen: (open: boolean) => void;
  /** Merge typed parameter values into a node's `data.params` (F2-4). */
  updateNodeParams: (nodeId: string, patch: ParamPatch) => void;
  /** Edit the legacy free-text `data.prompt` field (F2-4). */
  updateNodePrompt: (nodeId: string, prompt: string) => void;
  /** Replace the whole graph (used by workflow persistence on load). */
  hydrate: (nodes: Node[], edges: FlowEdge[], workflowId: string, name: string) => void;
};

/* ───── Demo graph (matches ref.html layout, kept here as initial state) ───── */

/* ───── Demo graph (matches the reference layout) ─────
   Topology: Model ─┐
                    ├─→ Sampler ─→ Image
   Positive ────────┤
   Negative ────────┘
   Edges carry an optional `label` + `tone` for the midpoint pill tag. */

const initialNodes: Node[] = [
  {
    id: "model",
    type: "model",
    position: { x: 60, y: 220 },
    data: {
      title: "Model",
      tone: "#7928ca",
      outputs: [
        { id: "model", kind: "model", label: "model", dotColor: "#f5a623" },
        { id: "positive", kind: "conditioning", label: "positive", dotColor: "#50e3c2" },
        { id: "negative", kind: "conditioning", label: "negative", dotColor: "#ff0080" },
      ],
      parameters: [
        {
          id: "checkpoint",
          label: "",
          value: "sdxl_base_1.0.safetensors",
          kind: "select",
          options: [
            "sdxl_base_1.0.safetensors",
            "dreamshaper_8.safetensors",
            "rev_animated.safetensors",
          ],
        },
      ],
    },
  },
  {
    id: "positive",
    type: "prompt",
    position: { x: 380, y: 60 },
    data: {
      title: "Positive",
      tone: "#50e3c2",
      prompt:
        "A black bear with a pink snout, minimalist style, soft gradients, clear blue sky",
    },
  },
  {
    id: "negative",
    type: "prompt",
    position: { x: 380, y: 320 },
    data: {
      title: "Negative",
      tone: "#ff0080",
      prompt:
        "No text, unnecessary details, background objects, other animals or people",
    },
  },
  {
    id: "image-generator",
    type: "imageGenerator",
    position: { x: 720, y: 120 },
    data: {
      title: "Sampler",
      tone: "#7928ca",
      inputs: [
        { id: "model", kind: "model", label: "model", dotColor: "#f5a623" },
        { id: "positive", kind: "conditioning", label: "positive", dotColor: "#50e3c2" },
        { id: "negative", kind: "conditioning", label: "negative", dotColor: "#ff0080" },
        { id: "latent", kind: "latent", label: "latent", dotColor: "#7928ca" },
      ],
      outputs: [
        { id: "image", kind: "image", label: "image", dotColor: "#50e3c2" },
      ],
      parameters: [
        { id: "seed", label: "Seed", value: "12345", tag: "Fixed", kind: "int" },
        { id: "steps", label: "Steps", value: "30", kind: "int" },
        { id: "cfg", label: "CFG scale", value: "8.0", kind: "float", min: 1, max: 20 },
        {
          id: "sampler",
          label: "Sampler",
          value: "dpm++ 2M",
          kind: "select",
          options: ["euler", "euler a", "dpm++ 2M", "dpm++ SDE", "ddim"],
        },
        {
          id: "scheduler",
          label: "Scheduler",
          value: "karras",
          kind: "select",
          options: ["normal", "karras", "exponential", "sgm_uniform"],
        },
      ],
    },
  },
  {
    id: "image",
    type: "imageOutput",
    position: { x: 1080, y: 140 },
    data: {
      title: "Image",
      tone: "#50e3c2",
      inputs: [
        { id: "image", kind: "image", label: "image", dotColor: "#50e3c2" },
      ],
    },
  },
];

const initialEdges: FlowEdge[] = [
  {
    id: "e-model",
    source: "model",
    sourceHandle: "model",
    target: "image-generator",
    targetHandle: "model",
    type: "flow",
    data: {
      sourceKind: "model",
      targetKind: "model",
    },
  },
  {
    id: "e-image",
    source: "image-generator",
    sourceHandle: "image",
    target: "image",
    targetHandle: "image",
    type: "flow",
    data: {
      sourceKind: "image",
      targetKind: "image",
    },
  },
];

/**
 * Workflow store — single source of truth for editor state.
 *
 * Wrapped in `zundo`'s `temporal` middleware:
 *   - `nodes` and `edges` mutations are recorded (drag, add, delete, connect)
 *   - `selectedNode` is excluded from history (selection bounce is noise)
 *   - action functions are not part of the persisted/tracked shape
 */
export const useWorkflowStore = create<WorkflowState>()(
  temporal(
    (set, get) => {
      const initial = {
        id: "main",
        name: "Untitled Workflow",
        nodes: initialNodes,
        edges: initialEdges,
        selectedNode: null,
        propertiesPanelOpen: false,
        onNodesChange: (changes: NodeChange[]) => {
          const s = get();
          set({ nodes: applyNodeChanges(changes, s.nodes) });
        },
        onEdgesChange: (changes: EdgeChange[]) => {
          const s = get();
          set({ edges: applyEdgeChanges(changes, s.edges) as FlowEdge[] });
        },
        onConnect: (conn: Connection) => {
          const s = get();
          const data: FlowEdgeData = {
            sourceKind: deriveKind(s.nodes, conn.source, conn.sourceHandle),
            targetKind: deriveKind(s.nodes, conn.target, conn.targetHandle),
          };
          const newEdges = rfAddEdge(
            { ...conn, type: "flow", data },
            s.edges,
          ) as unknown as FlowEdge[];
          set({ edges: newEdges });
        },
        onNodeSelect: (sel: SelectionInfo) =>
          set(
            sel
              ? { selectedNode: sel, propertiesPanelOpen: true }
              : { selectedNode: sel, propertiesPanelOpen: false },
          ),
        setPropertiesPanelOpen: (open: boolean) =>
          set({ propertiesPanelOpen: open }),
        updateNodeParams: (nodeId: string, patch: ParamPatch) =>
          set((s) => ({
            nodes: s.nodes.map((node) => {
              if (node.id !== nodeId) return node;
              const existing =
                (node.data as { params?: ParamPatch } | undefined)?.params ??
                {};
              return {
                ...node,
                data: {
                  ...node.data,
                  params: { ...existing, ...patch },
                },
              };
            }),
          })),
        updateNodePrompt: (nodeId: string, prompt: string) =>
          set((s) => ({
            nodes: s.nodes.map((node) =>
              node.id === nodeId
                ? { ...node, data: { ...node.data, prompt } }
                : node,
            ),
          })),
        hydrate: (nodes: Node[], edges: FlowEdge[], workflowId: string, name: string) =>
          set({ nodes, edges, id: workflowId, name }),
      };
      return initial;
    },
    {
      partialize: (state) => ({
        nodes: state.nodes,
        edges: state.edges,
      }),
      limit: 100,
      equality: (a, b) => a.nodes === b.nodes && a.edges === b.edges,
    },
  ),
);

/* ───── Hooks ───── */

/** Typed accessor for the temporal (undo/redo) slice. */
export const useWorkflowTemporal = <T,>(
  selector: (state: TemporalState<Pick<WorkflowState, "nodes" | "edges">>) => T,
): T => useStore(useWorkflowStore.temporal, selector);

/* ───── Imperative helpers (for non-React callers) ───── */

export const onNodeSelect = (s: SelectionInfo) =>
  useWorkflowStore.getState().onNodeSelect(s);

/* ───── Local ───── */

/** Resolve a socket's `kind` from a node id + handle id, for new connections. */
function deriveKind(
  nodes: Node[],
  nodeId: string | null,
  handleId: string | null,
): string {
  if (!nodeId) return "latent";
  const node = nodes.find((n) => n.id === nodeId);
  if (!node) return "latent";
  const data = node.data as { inputs?: { id: string; kind: string }[]; outputs?: { id: string; kind: string }[] } | undefined;
  if (!data) return "latent";
  const pool = [...(data.inputs ?? []), ...(data.outputs ?? [])];
  const sock = pool.find((s) => s.id === handleId);
  return sock?.kind ?? "latent";
}
