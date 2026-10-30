import type { Viewport } from "@xyflow/react";
import { LAYOUT } from "@/design/tokens";

/**
 * Insets AppShell gives the surface holding the topbar and the routed area.
 * Mirrors the `pt-2 pr-2 pb-2` (and `pl-2` when the sidebar is hidden) classes.
 */
const SHELL_INSET = 8;

export type CanvasAreaInput = {
  windowWidth: number;
  windowHeight: number;
  sidebarCollapsed: boolean;
  panelOpen: boolean;
};

export type CanvasAreaLayout = {
  /** Canvas width after the sidebar and the AppShell insets, in px. */
  width: number;
  /** Canvas height after the AppShell insets and the 48px topbar, in px. */
  height: number;
  /** Width the floating session panel covers, including both 16px gaps. */
  coverage: number;
  /** Canvas width the session panel leaves visible; never below 1. */
  visibleWidth: number;
};

/**
 * The canvas rectangle the workbench shell leaves for the board, plus the
 * strip a floating session panel leaves visible. The canvas is not rendered
 * while the library is open, so the shared window/sidebar/panel state is the
 * only way a library action can know where its visible center is.
 */
export function canvasAreaLayout({
  windowWidth,
  windowHeight,
  sidebarCollapsed,
  panelOpen,
}: CanvasAreaInput): CanvasAreaLayout {
  const sidebar = sidebarCollapsed ? 0 : LAYOUT.sidebar;
  // The sidebar supplies the left edge when open; the shell adds its own 8px
  // inset only once the sidebar is gone.
  const leftInset = sidebarCollapsed ? SHELL_INSET : 0;
  const width = Math.max(1, windowWidth - sidebar - SHELL_INSET - leftInset);
  const height = Math.max(1, windowHeight - SHELL_INSET * 2 - LAYOUT.topbar);
  const coverage = panelOpen ? LAYOUT.panel + LAYOUT.panelGap * 2 : 0;
  return {
    width,
    height,
    coverage,
    visibleWidth: Math.max(1, width - coverage),
  };
}

/**
 * Flow coordinate rendered at the center of the canvas strip the session
 * panel leaves visible. Inverse of the pan/zoom transform: screen = flow *
 * zoom + offset.
 */
export function placementCenter(
  viewport: Viewport,
  area: Pick<CanvasAreaLayout, "visibleWidth" | "height">,
): { x: number; y: number } {
  const zoom = viewport.zoom > 0 ? viewport.zoom : 1;
  return {
    x: (area.visibleWidth / 2 - viewport.x) / zoom,
    y: (area.height / 2 - viewport.y) / zoom,
  };
}
