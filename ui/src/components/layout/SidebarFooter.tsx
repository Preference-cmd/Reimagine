import { Settings, User } from "lucide-react";
import { useUIStore } from "@/store/uiStore";

/**
 * Sidebar footer — user card + settings button.
 * Compact user info at bottom with settings gear.
 */
export function SidebarFooter() {
  const setActiveSection = useUIStore((s) => s.setActiveSidebarSection);

  return (
    <div className="border-t border-outline/30 px-3 py-3">
      <div className="flex items-center gap-2.5">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-surface-container-high text-on-surface-variant">
          <User className="h-4 w-4" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="truncate text-body-sm font-medium text-on-surface">User</div>
          <div className="truncate text-caption text-on-surface-variant/70">Local workspace</div>
        </div>
        <button
          type="button"
          aria-label="Settings"
          onClick={() => setActiveSection("settings")}
          className="flex h-7 w-7 shrink-0 cursor-pointer items-center justify-center rounded-lg text-on-surface-variant transition-colors duration-150 hover:bg-control-hover hover:text-on-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30"
        >
          <Settings className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
