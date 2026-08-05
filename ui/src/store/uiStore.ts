import { create } from "zustand";
import type { XYPosition } from "@xyflow/react";
import type { NodeDef } from "@/ipc/schemas";

/**
 * UI shell state (F3-1/F3-3): active panel, command palette mode and
 * position, the floating context menu, and the rename dialog target.
 * View-only state — deliberately kept out of the workflow (undo) store.
 */

export type PaletteMode =
  | { kind: "closed" }
  | { kind: "command" }
  /** Node-type picker; `position` is the flow coordinate to insert at. */
  | { kind: "node"; position: XYPosition };

export type ContextMenuItem = {
  id: string;
  label: string;
  danger?: boolean;
  disabled?: boolean;
  shortcut?: string;
  onSelect: () => void;
};

export type ContextMenuState = {
  x: number;
  y: number;
  items: ContextMenuItem[];
} | null;

export type RenameTarget = { id: string; title: string } | null;

type UIState = {
  activePanel: string | null;
  setActivePanel: (panel: string | null) => void;
  togglePanel: (panel: string) => void;

  palette: PaletteMode;
  openCommandPalette: () => void;
  /** Open the node picker with nodes inserted at `position` (flow coords). */
  openNodePalette: (position: XYPosition) => void;
  closePalette: () => void;

  contextMenu: ContextMenuState;
  openContextMenu: (
    x: number,
    y: number,
    items: ContextMenuItem[],
  ) => void;
  closeContextMenu: () => void;

  renameTarget: RenameTarget;
  startRename: (target: { id: string; title: string }) => void;
  finishRename: () => void;
};

export const useUIStore = create<UIState>()((set) => ({
  activePanel: null,
  setActivePanel: (panel) => set({ activePanel: panel }),
  togglePanel: (panel) =>
    set((state) => ({ activePanel: state.activePanel === panel ? null : panel })),

  palette: { kind: "closed" },
  openCommandPalette: () =>
    set((state) => ({
      palette: state.palette.kind === "command" ? { kind: "closed" } : { kind: "command" },
    })),
  openNodePalette: (position) => set({ palette: { kind: "node", position } }),
  closePalette: () => set({ palette: { kind: "closed" } }),

  contextMenu: null,
  openContextMenu: (x, y, items) => set({ contextMenu: { x, y, items } }),
  closeContextMenu: () => set({ contextMenu: null }),

  renameTarget: null,
  startRename: (target) => set({ renameTarget: target }),
  finishRename: () => set({ renameTarget: null }),
}));

/** Selector: node defs grouped by category, preserving catalog order (F2-2). */
export function groupDefsByCategory(defs: NodeDef[]): Array<[string, NodeDef[]]> {
  const groups: Array<[string, NodeDef[]]> = [];
  const index = new Map<string, number>();
  for (const def of defs) {
    const category = def.category || "Other";
    const existing = index.get(category);
    if (existing === undefined) {
      index.set(category, groups.length);
      groups.push([category, [def]]);
    } else {
      groups[existing][1].push(def);
    }
  }
  return groups;
}
