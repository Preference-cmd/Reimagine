import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import { useUIStore } from "@/store/uiStore";

/**
 * Floating right-click menu (F3-1).
 *
 * State-driven from the UI store (single menu instance for canvas, node
 * and edge targets — React Flow reports the target via its pane/node/edge
 * context-menu handlers). Rendered once in AppShell, above all overlays.
 */
export function ContextMenuPanel() {
  const menu = useUIStore((s) => s.contextMenu);
  const close = useUIStore((s) => s.closeContextMenu);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!menu) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
    };
    const onPointerDown = (event: PointerEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        close();
      }
    };
    const onScroll = () => close();
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("pointerdown", onPointerDown, true);
    window.addEventListener("wheel", onScroll, { passive: true, capture: true });
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("pointerdown", onPointerDown, true);
      window.removeEventListener("wheel", onScroll, { capture: true });
    };
  }, [close, menu]);

  if (!menu) return null;

  return (
    <div
      ref={menuRef}
      role="menu"
      className="panel-raised pointer-events-auto fixed z-[var(--overlay-z-modal)] w-48 overflow-hidden rounded-lg py-1"
      style={{ left: menu.x, top: menu.y }}
    >
      {menu.items.map((item) => (
        <button
          key={item.id}
          type="button"
          role="menuitem"
          disabled={item.disabled}
          onClick={() => {
            close();
            item.onSelect();
          }}
          className={cn(
            "flex w-full cursor-pointer items-center gap-2 px-3 py-1.5 text-left text-caption font-medium text-on-surface transition-colors hover:bg-control-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30 disabled:cursor-default disabled:opacity-45",
            item.danger && "text-destructive hover:bg-destructive/10",
          )}
        >
          <span className="flex-1 truncate">{item.label}</span>
          {item.shortcut && (
            <span className="text-caption text-on-surface-variant/70">{item.shortcut}</span>
          )}
        </button>
      ))}
    </div>
  );
}
