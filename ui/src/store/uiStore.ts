import { create } from "zustand";
import type { Viewport, XYPosition } from "@xyflow/react";
import type { ActivityActor, NodeDef } from "@/ipc/schemas";

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

/**
 * Canvas highlight requested by hovering an activity record. Items whose id is
 * in `targets` keep full ink plus a ring; every other item fades. `actorKind`
 * chooses the ring colour: agent gold for Agent, accent for everyone else.
 */
export type CanvasTrace = {
  targets: string[];
  actorKind: ActivityActor["kind"];
};

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

  // ── Canvas viewports ────────────────────────────────────
  /**
   * React Flow viewport per project, persisted by the canvas on every pan or
   * zoom end. The canvas restores it on mount and on project switches so a
   * canvas → library → canvas trip keeps the view, and the library derives the
   * placement center from it.
   */
  canvasViewports: Record<string, Viewport>;
  setCanvasViewport: (projectId: string, viewport: Viewport) => void;

  // ── Canvas trace ────────────────────────────────────────
  /** Hover trace from the activity panel; null when nothing is hovered. */
  canvasTrace: CanvasTrace | null;
  setCanvasTrace: (trace: CanvasTrace) => void;
  clearCanvasTrace: () => void;

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

  // Canvas viewports
  canvasViewports: {},
  setCanvasViewport: (projectId, viewport) =>
    set((state) => ({
      canvasViewports: { ...state.canvasViewports, [projectId]: viewport },
    })),

  // Canvas trace
  canvasTrace: null,
  setCanvasTrace: (trace) => set({ canvasTrace: trace }),
  clearCanvasTrace: () => set({ canvasTrace: null }),

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
