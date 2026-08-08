import { Settings } from "lucide-react";
import { useUIStore } from "@/store/uiStore";
import * as m from "$paraglide/messages";

/**
 * Sidebar footer — settings button with optional progress indicator.
 */
export function SidebarFooter() {
  const setActiveSection = useUIStore((s) => s.setActiveSidebarSection);
  const progress = useUIStore((s) => s.sidebarProgress);

  return (
    <div className="border-t border-sidebar-border px-2 pb-2 pt-2">
      <div className="flex items-center justify-between">
        <button
          type="button"
          aria-label={m["sidebar.settings"]()}
          onClick={() => setActiveSection("settings")}
          className="flex h-8 cursor-pointer items-center gap-2.5 rounded-md px-2.5 text-left text-sm text-sidebar-text-secondary transition-colors hover:bg-sidebar-item-hover hover:text-sidebar-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/20"
        >
          <Settings className="h-4 w-4 shrink-0 text-sidebar-text-muted" />
          <span className="truncate">{m["sidebar.settings"]()}</span>
        </button>

        {progress > 0 && (
          <span className="rounded-full bg-status-success/20 px-2 py-0.5 text-caption font-medium tabular-nums text-status-success">
            {progress}%
          </span>
        )}
      </div>
    </div>
  );
}
