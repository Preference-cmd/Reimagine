import type { XYPosition } from "@xyflow/react";
import { useWorkflowStore } from "@/store/workflow";

/**
 * React Flow instance registry (F2-2/F3-3).
 *
 * The canvas instance is needed outside the React tree — the command
 * palette, explorer drag-and-drop and context menus convert screen/cursor
 * positions into flow coordinates. The instance is stable for the app's
 * lifetime, so a module singleton is sufficient (no reactivity needed).
 */

type ReactFlowInstanceLike = {
  screenToFlowPosition: (client: XYPosition) => XYPosition;
};

let instance: ReactFlowInstanceLike | null = null;

export function registerFlowInstance(
  next: ReactFlowInstanceLike | null,
): void {
  instance = next;
}

/**
 * Flow position at the center of the viewport (used by "Add Node" in the
 * command palette). Falls back to a fixed offset when no instance exists
 * (e.g. before the canvas mounts or in tests).
 */
export function flowViewportCenter(): XYPosition {
  if (!instance) return { x: 120, y: 120 };
  return instance.screenToFlowPosition({
    x: window.innerWidth / 2,
    y: window.innerHeight / 2,
  });
}

/** Flow position for a screen/client point (cursor, drop point). */
export function flowPositionFor(client: XYPosition): XYPosition {
  if (!instance) return { x: client.x, y: client.y };
  return instance.screenToFlowPosition(client);
}

/** Select-all via store changes (no instance method exists in RF v12). */
export function selectAllNodes(): void {
  const nodes = useWorkflowStore.getState().nodes;
  useWorkflowStore
    .getState()
    .onNodesChange(
      nodes.map((node) => ({ id: node.id, type: "select", selected: true })),
    );
}
