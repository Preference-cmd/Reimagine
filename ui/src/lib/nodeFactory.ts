import type { Node, XYPosition } from "@xyflow/react";
import { categoryTone, normalizeParamValue, type ParamValue } from "@/lib/nodes";
import type { NodeDef } from "@/ipc/schemas";
import { useNodeRegistryStore, selectNodeDef } from "@/store/nodeRegistry";
import { onNodeSelect, useWorkflowStore } from "@/store/workflow";

/**
 * Node creation (F2-2): build React Flow nodes from catalog `NodeDef`s with
 * default parameter values, plus the imperative `createNodeAt` used by the
 * explorer drag-drop, canvas double-click and command palette.
 */

export type CatalogNodeData = {
  title: string;
  tone: string;
  params: Record<string, ParamValue>;
  disabled?: boolean;
};

/** dataTransfer mime type for drag-and-drop node creation (F2-2). */
export const NODE_DRAG_MIME = "application/x-reimagine-node";

/** Default parameter values from a def, normalized to frontend `ParamValue`s. */
export function defaultParamsFor(def: NodeDef): Record<string, ParamValue> {
  const params: Record<string, ParamValue> = {};
  for (const spec of def.parameters) {
    const value = normalizeParamValue(spec.default);
    if (value !== undefined) params[spec.id] = value;
  }
  return params;
}

/** Build a React Flow node for a catalog def (title/tone for legacy renderers). */
export function createNodeFromDef(
  def: NodeDef,
  position: XYPosition,
  id: string,
): Node<CatalogNodeData> {
  return {
    id,
    type: def.type,
    position,
    data: {
      title: def.displayName,
      tone: categoryTone(def.category),
      params: defaultParamsFor(def),
    },
  };
}

/** Collision-free id for a new node of a given catalog type. */
export function uniqueNodeId(existing: Array<{ id: string }>, type: string): string {
  const base = type.replace(/[^a-zA-Z0-9_-]/g, "-").replace(/^-+|-+$/g, "");
  const prefix = base || "node";
  let candidate = prefix;
  let index = 1;
  const taken = new Set(existing.map((node) => node.id));
  while (taken.has(candidate)) {
    candidate = `${prefix}-${index}`;
    index += 1;
  }
  return candidate;
}

/**
 * Create a node from the catalog at a flow position and select it.
 * Returns false (without mutating) when the type is not in the catalog.
 */
export function createNodeAt(typeId: string, position: XYPosition): boolean {
  const def = selectNodeDef(useNodeRegistryStore.getState().defs, typeId);
  if (!def) return false;
  const id = uniqueNodeId(useWorkflowStore.getState().nodes, def.type);
  const node = createNodeFromDef(def, position, id);
  useWorkflowStore.getState().addNode(node);
  onNodeSelect({ id, type: def.type });
  return true;
}

/** Short display name for a catalog type (used in menus and toasts). */
export function displayNameFor(defs: Map<string, NodeDef>, typeId: string): string {
  return defs.get(typeId)?.displayName ?? typeId;
}
