import { SidebarHeader } from "./SidebarHeader";
import { SidebarNav } from "./SidebarNav";
import { SidebarSettingsNav } from "./SidebarSettingsNav";
import { RecentWorkflows } from "./RecentWorkflows";
import { SidebarFooter } from "./SidebarFooter";
import { useUIStore } from "@/store/uiStore";

/**
 * Sidebar — fixed-width left column (280 px).
 *
 * Normal mode: Logo + Nav + Recent + User Card
 * Settings mode: Back button + Search + Settings nav groups
 */
export function Sidebar() {
  const activeSection = useUIStore((s) => s.activeSidebarSection);
  const isSettings = activeSection === "settings";

  return (
    <aside className="sidebar-root flex h-full w-[280px] shrink-0 flex-col border-r border-outline/60 bg-surface-dim">
      {isSettings ? (
        <SidebarSettingsNav />
      ) : (
        <>
          <SidebarHeader />
          <SidebarNav />
          <div className="min-h-0 flex-1 overflow-y-auto scrollbar-hide">
            <RecentWorkflows />
          </div>
          <SidebarFooter />
        </>
      )}
    </aside>
  );
}
