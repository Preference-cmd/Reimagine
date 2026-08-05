import type { Node } from "@xyflow/react";
import type { FlowEdge } from "@/components/canvas/FlowEdge";

/**
 * Workflow persistence codec (F1-2).
 *
 * Converts between the React Flow editor graph (nodes/edges with positions
 * and UI data) and the backend `Workflow` JSON persisted by the Rust
 * `WorkflowService` (logical nodes + typed params + layout).
 *
 * Save (frontend → Rust):
 *   - node `type` → `type_id`, `position` → `layout.nodes`
 *   - node `data.title` → `label`, `data.parameters` → typed `params`
 *   - the full UI payload (`data`) is preserved verbatim in a reserved
 *     `ui_data` string param so loads roundtrip losslessly
 * Load (Rust → frontend):
 *   - `layout.nodes[id]` → `position`, `type_id` → `type`
 *   - `ui_data` (if present) restores the exact UI payload
 */

export const WORKFLOW_SCHEMA_VERSION = "reimagine.workflow.v1";

/** Reserved param slot carrying the lossless UI payload. */
const UI_DATA_SLOT = "ui_data";

type ParamValue = {
  type: "integer" | "float" | "bool" | "select" | "string";
  value: string | number | boolean;
};

export type BackendWorkflow = {
  schema_version: string;
  id: string;
  version: number;
  metadata: { name?: string; description?: string; created_by?: string };
  interface: { inputs: unknown[]; outputs: unknown[] };
  nodes: Array<{
    id: string;
    type_id: string;
    label?: string | null;
    params?: Record<string, ParamValue>;
  }>;
  edges: Array<{
    id: string;
    from: { node: string; slot: string };
    to: { node: string; slot: string };
  }>;
  layout: {
    nodes?: Record<string, { x: number; y: number }>;
    viewport?: { x: number; y: number; zoom: number };
  };
};

export type WorkflowGraph = {
  nodes: Node[];
  edges: FlowEdge[];
  name: string;
};

/** Serialize the editor graph into a backend `Workflow` payload. */
export function workflowToJson(
  nodes: Node[],
  edges: FlowEdge[],
  id: string,
  name: string,
): BackendWorkflow {
  return {
    schema_version: WORKFLOW_SCHEMA_VERSION,
    id,
    version: 1,
    metadata: { name },
    interface: { inputs: [], outputs: [] },
    nodes: nodes.map((node) => ({
      id: node.id,
      type_id: node.type ?? "builtin.unknown",
      ...(typeof node.data?.title === "string"
        ? { label: node.data.title }
        : {}),
      params: {
        ...paramsFromData(node.data),
        [UI_DATA_SLOT]: { type: "string", value: JSON.stringify(node.data ?? {}) },
      },
    })),
    edges: edges.filter(isPersistableEdge).map((edge) => ({
      id: edge.id,
      from: { node: edge.source, slot: edge.sourceHandle },
      to: { node: edge.target, slot: edge.targetHandle },
    })),
    layout: {
      nodes: Object.fromEntries(
        nodes.map((node) => [
          node.id,
          { x: node.position.x, y: node.position.y },
        ]),
      ),
    },
  };
}

/** Restore the editor graph from a backend `Workflow` payload. */
export function workflowFromJson(json: unknown): WorkflowGraph {
  const workflow = json as Partial<BackendWorkflow> | null;
  if (!workflow || typeof workflow !== "object") {
    throw new Error("invalid workflow json: not an object");
  }

  const layoutNodes = workflow.layout?.nodes ?? {};
  const nodes: Node[] = (workflow.nodes ?? []).map((node) => {
    const position = layoutNodes[node.id] ?? { x: 0, y: 0 };
    let data: Record<string, unknown> = {
      title: node.label ?? node.type_id,
    };
    const uiData = node.params?.[UI_DATA_SLOT];
    if (uiData && typeof uiData.value === "string") {
      try {
        const parsed: unknown = JSON.parse(uiData.value);
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          data = { ...data, ...(parsed as Record<string, unknown>) };
        }
      } catch {
        // Malformed ui_data — fall back to label/params reconstruction.
      }
    }
    return { id: node.id, type: node.type_id, position, data };
  });

  const edges: FlowEdge[] = (workflow.edges ?? [])
    .filter(
      (edge) =>
        edge?.from?.node && edge?.from?.slot && edge?.to?.node && edge?.to?.slot,
    )
    .map((edge) => ({
      id: edge.id,
      source: edge.from.node,
      sourceHandle: edge.from.slot,
      target: edge.to.node,
      targetHandle: edge.to.slot,
      type: "flow",
      data: {
        sourceKind: socketKind(nodes, edge.from.node, edge.from.slot),
        targetKind: socketKind(nodes, edge.to.node, edge.to.slot),
      },
    }));

  return {
    nodes,
    edges,
    name: workflow.metadata?.name ?? "Untitled Workflow",
  };
}

/* ───── Local ───── */

type EdgeWithHandles = FlowEdge & { sourceHandle: string; targetHandle: string };

function isPersistableEdge(edge: FlowEdge): edge is EdgeWithHandles {
  return Boolean(edge.source && edge.sourceHandle && edge.target && edge.targetHandle);
}

function paramsFromData(
  data: Record<string, unknown> | undefined,
): Record<string, ParamValue> {
  const rows = Array.isArray(data?.parameters)
    ? (data.parameters as Array<{ id?: unknown; value?: unknown }>)
    : [];
  const params: Record<string, ParamValue> = {};
  for (const row of rows) {
    if (typeof row.id !== "string" || row.id.length === 0) continue;
    const value =
      typeof row.value === "string" ? row.value : String(row.value ?? "");
    params[row.id] = inferParamValue(value);
  }
  return params;
}

function inferParamValue(value: string): ParamValue {
  if (/^-?\d+$/.test(value)) {
    return { type: "integer", value: parseInt(value, 10) };
  }
  if (/^-?\d*\.\d+$/.test(value)) {
    return { type: "float", value: parseFloat(value) };
  }
  if (value === "true") return { type: "bool", value: true };
  if (value === "false") return { type: "bool", value: false };
  return { type: "select", value };
}

/** Resolve a socket `kind` from a node id + handle id, for restored edges. */
function socketKind(
  nodes: Node[],
  nodeId: string,
  handleId: string,
): string {
  const node = nodes.find((n) => n.id === nodeId);
  const data = node?.data as
    | { inputs?: { id: string; kind: string }[]; outputs?: { id: string; kind: string }[] }
    | undefined;
  const pool = [...(data?.inputs ?? []), ...(data?.outputs ?? [])];
  return pool.find((socket) => socket.id === handleId)?.kind ?? "latent";
}
