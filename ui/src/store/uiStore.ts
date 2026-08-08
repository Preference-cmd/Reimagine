import { create } from "zustand";
import type { XYPosition } from "@xyflow/react";
import type { NodeDef } from "@/ipc/schemas";

/**
 * UI shell state — sidebar navigation, properties drawer,
 * command palette, context menu, and rename dialog.
 * View-only state — deliberately kept out of the workflow (undo) store.
 */

export type SidebarSection =
  | "chat"
  | "new-task"
  | "pull-requests"
  | "scheduled"
  | "plugins"
  | "workflows"
  | "models"
  | "runs"
  | "assets"
  | "settings"
  | null;

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
  // ── Sidebar ─────────────────────────────────────────────
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;
  activeSidebarSection: SidebarSection;
  setActiveSidebarSection: (section: SidebarSection) => void;

  // ── Sidebar projects ────────────────────────────────────
  projectsExpanded: boolean;
  setProjectsExpanded: (expanded: boolean) => void;
  activeProjectId: string | null;
  setActiveProjectId: (id: string | null) => void;

  // ── Sidebar progress ────────────────────────────────────
  sidebarProgress: number;
  setSidebarProgress: (progress: number) => void;

  // ── Properties drawer ───────────────────────────────────
  propertiesDrawerOpen: boolean;
  setPropertiesDrawerOpen: (open: boolean) => void;

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
  // Sidebar
  sidebarCollapsed: false,
  toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
  activeSidebarSection: "chat" as SidebarSection,
  setActiveSidebarSection: (section) => set({ activeSidebarSection: section }),

  // Sidebar projects
  projectsExpanded: true,
  setProjectsExpanded: (expanded) => set({ projectsExpanded: expanded }),
  activeProjectId: null,
  setActiveProjectId: (id) => set({ activeProjectId: id }),

  // Sidebar progress
  sidebarProgress: 0,
  setSidebarProgress: (progress) => set({ sidebarProgress: progress }),

  // Properties drawer
  propertiesDrawerOpen: false,
  setPropertiesDrawerOpen: (open: boolean) => set({ propertiesDrawerOpen: open }),

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
