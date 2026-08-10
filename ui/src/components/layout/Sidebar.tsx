import { useCallback, useRef, useState } from "react";
import { SidebarHeader } from "./SidebarHeader";
import { SidebarNav } from "./SidebarNav";
import { SidebarFooter } from "./SidebarFooter";
import { SidebarSettingsNav } from "./SidebarSettingsNav";
import { RecentWorkflows } from "./RecentWorkflows";
import { useLocation } from "@tanstack/react-router";
import { useUIStore } from "@/store/uiStore";

const MIN_WIDTH = 120;
const MAX_WIDTH = 400;
const COLLAPSED_WIDTH = 48;
const DBLCLICK_THRESHOLD = 250;

/**
 * Sidebar — supports drag-to-resize via mouse handle on right edge.
 * Width persisted in uiStore.sidebarWidth (localStorage-backed).
 * Double-click handle to collapse/expand.
 */
export function Sidebar() {
  const { pathname } = useLocation();
  const isSettings = pathname === "/settings";
  const width = useUIStore((s) => s.sidebarWidth);
  const setWidth = useUIStore((s) => s.setSidebarWidth);
  const dragging = useRef(false);
  const startX = useRef(0);
  const startWidth = useRef(0);
  const lastClickTime = useRef(0);
  const [isDragging, setIsDragging] = useState(false);

  const onPointerDown = useCallback(
    (e: React.PointerEvent) => {
      // Double-click to toggle collapse/expand
      const now = Date.now();
      if (now - lastClickTime.current < DBLCLICK_THRESHOLD) {
        lastClickTime.current = 0;
        setWidth(width > COLLAPSED_WIDTH + 10 ? COLLAPSED_WIDTH : 220);
        return;
      }
      lastClickTime.current = now;

      e.preventDefault();
      dragging.current = true;
      setIsDragging(true);
      startX.current = e.clientX;
      startWidth.current = width;
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";

      const onPointerMove = (ev: PointerEvent) => {
        if (!dragging.current) return;
        const delta = ev.clientX - startX.current;
        const next = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, startWidth.current + delta));
        setWidth(next);
      };

      const onPointerUp = () => {
        dragging.current = false;
        setIsDragging(false);
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
        document.removeEventListener("pointermove", onPointerMove);
        document.removeEventListener("pointerup", onPointerUp);
      };

      document.addEventListener("pointermove", onPointerMove);
      document.addEventListener("pointerup", onPointerUp);
    },
    [width, setWidth],
  );

  return (
    <aside
      className="sidebar-root relative flex h-full shrink-0 flex-col bg-sidebar-bg"
      style={{
        width,
        transition: isDragging ? "none" : "width 150ms ease-out",
        boxShadow: "inset -1px 0 0 0 var(--color-sidebar-border)",
      }}
    >
      {isSettings ? (
        <SidebarSettingsNav />
      ) : (
        <>
          <SidebarHeader collapsed={width <= COLLAPSED_WIDTH + 10} />
          <SidebarNav collapsed={width <= COLLAPSED_WIDTH + 10}>
            <RecentWorkflows collapsed={width <= COLLAPSED_WIDTH + 10} />
          </SidebarNav>
          <SidebarFooter collapsed={width <= COLLAPSED_WIDTH + 10} />
        </>
      )}

      {/* Drag handle — right edge, 6px grab zone */}
      <div
        onPointerDown={onPointerDown}
        className={`absolute right-0 top-0 z-10 h-full w-1.5 cursor-col-resize transition-colors ${
          isDragging ? "bg-primary/60" : "bg-transparent hover:bg-primary/30"
        }`}
        aria-label="Resize sidebar"
        role="separator"
        aria-orientation="vertical"
        aria-valuenow={width}
        aria-valuemin={MIN_WIDTH}
        aria-valuemax={MAX_WIDTH}
      />
    </aside>
  );
}
