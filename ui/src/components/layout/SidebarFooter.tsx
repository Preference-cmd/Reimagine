import { Settings, User } from "lucide-react";
import { useUIStore } from "@/store/uiStore";

/**
 * Sidebar footer — user card + settings button.
 * Codex-style: compact user info at bottom with settings gear.
 */
export function SidebarFooter() {
  const setActiveSection = useUIStore((s) => s.setActiveSidebarSection);

  return (
    <div className="border-t border-outline/50 px-3 py-2.5">
      <div className="flex items-center gap-2">
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-surface-container-high text-on-surface-variant">
          <User className="h-3.5 w-3.5" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="truncate text-caption font-medium text-on-surface">User</div>
          <div className="truncate text-caption text-on-surface-variant">Local workspace</div>
        </div>
        <button
          type="button"
          aria-label="Settings"
          onClick={() => setActiveSection("settings")}
          className="flex h-6 w-6 shrink-0 cursor-pointer items-center justify-center rounded-md text-on-surface-variant transition-colors duration-150 hover:bg-control-hover hover:text-on-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30"
        >
          <Settings className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  );
}
