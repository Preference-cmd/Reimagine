import { Settings } from "lucide-react";
import { Link, useLocation } from "@tanstack/react-router";
import { cn } from "@/lib/utils";
import { useUIStore } from "@/store/uiStore";
import * as m from "$paraglide/messages";

/**
 * Sidebar footer — settings button with optional progress indicator.
 */
export function SidebarFooter({ collapsed }: { collapsed?: boolean }) {
  const progress = useUIStore((s) => s.sidebarProgress);
  const { pathname } = useLocation();
  const isActive = pathname === "/settings";

  return (
    <div className="border-t border-sidebar-border px-2 pb-2 pt-2">
      <div className="flex items-center justify-between">
        <Link
          to="/settings"
          aria-current={isActive ? "page" : undefined}
          aria-label={collapsed ? m["sidebar.settings"]() : undefined}
          title={collapsed ? m["sidebar.settings"]() : undefined}
          className={cn(
            "flex h-8 cursor-pointer items-center gap-2.5 rounded-md px-2.5 text-left text-sm transition-colors duration-150",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/20",
            collapsed && "justify-center px-0",
            isActive
              ? "bg-sidebar-item-active text-sidebar-text-primary"
              : "text-sidebar-text-secondary hover:bg-sidebar-item-hover hover:text-sidebar-text-primary",
          )}
        >
          <Settings
            className={cn(
              "h-4 w-4 shrink-0",
              isActive ? "text-sidebar-text-secondary" : "text-sidebar-text-muted",
            )}
          />
          {!collapsed && <span className="truncate">{m["sidebar.settings"]()}</span>}
        </Link>

        {!collapsed && progress > 0 && (
          <span className="rounded-full bg-status-success/20 px-2 py-0.5 text-caption font-medium tabular-nums text-status-success">
            {progress}%
          </span>
        )}
      </div>
    </div>
  );
}
