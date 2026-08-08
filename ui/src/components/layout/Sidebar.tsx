import { SidebarHeader } from "./SidebarHeader";
import { SidebarNav } from "./SidebarNav";
import { SidebarSettingsNav } from "./SidebarSettingsNav";
import { SidebarFooter } from "./SidebarFooter";
import { useUIStore } from "@/store/uiStore";

/**
 * Sidebar — Codex-style (220 px) with icon + text navigation.
 *
 * Normal mode: Logo + Nav + Footer
 * Settings mode: Full-width settings panel
 */
export function Sidebar() {
  const activeSection = useUIStore((s) => s.activeSidebarSection);
  const isSettings = activeSection === "settings";

  if (isSettings) {
    return (
      <aside className="sidebar-root flex h-full w-[260px] shrink-0 flex-col border-r border-sidebar-border bg-sidebar-bg">
        <SidebarSettingsNav />
      </aside>
    );
  }

  return (
    <aside className="sidebar-root flex h-full w-[220px] shrink-0 flex-col border-r border-sidebar-border bg-sidebar-bg">
      <SidebarHeader />
      <SidebarNav />
      <SidebarFooter />
    </aside>
  );
}
