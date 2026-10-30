import { create } from "zustand";
import type { XYPosition } from "@xyflow/react";
import type { NodeDef } from "@/ipc/schemas";

/**
 * UI shell state — session panel, canvas focus, command palette,
 * context menu, and rename dialog.
 * View-only state — deliberately kept out of the workflow (undo) store.
 * Route navigation is handled by TanStack Router (URL-based).
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
  // ── Session panel ───────────────────────────────────────
  sessionPanelOpen: boolean;
  setSessionPanelOpen: (open: boolean) => void;
  toggleSessionPanel: () => void;

  // ── Canvas focus ────────────────────────────────────────
  /** Item the canvas should center on its next render; cleared once handled. */
  canvasFocusItemId: string | null;
  requestCanvasFocus: (id: string) => void;
  clearCanvasFocus: () => void;

  // ── Canvas center memory ────────────────────────────────
  /**
   * Flow coordinates at the center of the canvas strip the session panel
   * leaves visible, kept current by the canvas. The library places new items
   * here; null until the canvas has measured itself in this session.
   */
  lastCanvasCenter: { x: number; y: number } | null;
  setLastCanvasCenter: (center: { x: number; y: number }) => void;

  // ── Settings nav ────────────────────────────────────────
  settingsNavId: string | null;
  setSettingsNavId: (id: string | null) => void;

  // ── Command palette ─────────────────────────────────────
  palette: PaletteMode;
  openCommandPalette: () => void;
  openNodePalette: (position: XYPosition) => void;
  closePalette: () => void;

  // ── Context menu ────────────────────────────────────────
  contextMenu: ContextMenuState;
  openContextMenu: (x: number, y: number, items: ContextMenuItem[]) => void;
  closeContextMenu: () => void;

  // ── Rename dialog ───────────────────────────────────────
  renameTarget: RenameTarget;
  startRename: (target: { id: string; title: string }) => void;
  finishRename: () => void;
};

export const useUIStore = create<UIState>()((set) => ({
  // Session panel
  sessionPanelOpen: true,
  setSessionPanelOpen: (open: boolean) => set({ sessionPanelOpen: open }),
  toggleSessionPanel: () => set((state) => ({ sessionPanelOpen: !state.sessionPanelOpen })),

  // Canvas focus
  canvasFocusItemId: null,
  requestCanvasFocus: (id: string) => set({ canvasFocusItemId: id }),
  clearCanvasFocus: () => set({ canvasFocusItemId: null }),

  // Canvas center memory
  lastCanvasCenter: null,
  setLastCanvasCenter: (center) => set({ lastCanvasCenter: center }),

  // Settings nav
  settingsNavId: "general",
  setSettingsNavId: (id: string | null) => set({ settingsNavId: id }),

  // Command palette
  palette: { kind: "closed" },
  openCommandPalette: () =>
    set((state) => ({
      palette: state.palette.kind === "command" ? { kind: "closed" } : { kind: "command" },
    })),
  openNodePalette: (position) => set({ palette: { kind: "node", position } }),
  closePalette: () => set({ palette: { kind: "closed" } }),

  // Context menu
  contextMenu: null,
  openContextMenu: (x, y, items) => set({ contextMenu: { x, y, items } }),
  closeContextMenu: () => set({ contextMenu: null }),

  // Rename dialog
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
