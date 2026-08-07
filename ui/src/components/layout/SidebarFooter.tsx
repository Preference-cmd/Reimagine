import { Settings } from "lucide-react";
import { useUIStore } from "@/store/uiStore";
import * as m from "$paraglide/messages";

/**
 * Sidebar footer — settings button at bottom.
 */
export function SidebarFooter() {
  const setActiveSection = useUIStore((s) => s.setActiveSidebarSection);

  return (
    <div className="px-2 pb-2 pt-1">
      <button
        type="button"
        aria-label={m["sidebar.settings"]()}
        onClick={() => setActiveSection("settings")}
        className="flex h-8 w-full cursor-pointer items-center gap-2.5 rounded-md px-2.5 text-left text-sm text-white/40 transition-colors hover:bg-white/[0.05] hover:text-white/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/20"
      >
        <Settings className="h-4 w-4 shrink-0 text-white/30" />
        <span className="truncate">{m["sidebar.settings"]()}</span>
      </button>
    </div>
  );
}
