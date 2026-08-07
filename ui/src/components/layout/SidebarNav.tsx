import { Workflow, Boxes, History, Image, Settings } from "lucide-react";
import { cn } from "@/lib/utils";
import { useUIStore, type SidebarSection } from "@/store/uiStore";

type NavItem = {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  section: SidebarSection;
};

const NAV_ITEMS: NavItem[] = [
  { icon: Workflow, label: "Workflows", section: "workflows" },
  { icon: Boxes, label: "Models", section: "models" },
  { icon: History, label: "Runs", section: "runs" },
  { icon: Image, label: "Assets", section: "assets" },
];

export function SidebarNav() {
  const activeSection = useUIStore((s) => s.activeSidebarSection);
  const setActiveSection = useUIStore((s) => s.setActiveSidebarSection);

  return (
    <nav className="flex flex-col gap-0.5 px-2" aria-label="Sidebar navigation">
      {NAV_ITEMS.map((item) => {
        const Icon = item.icon;
        const active = activeSection === item.section;

        return (
          <button
            key={item.section}
            type="button"
            aria-current={active ? "page" : undefined}
            onClick={() => setActiveSection(item.section)}
            className={cn(
              "flex h-7 cursor-pointer items-center gap-2 rounded-md px-2 text-left text-body-sm font-medium transition-colors",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30",
              active
                ? "bg-surface-container-high text-on-surface"
                : "text-on-surface-variant hover:bg-control-hover hover:text-on-surface",
            )}
          >
            <Icon className="h-4 w-4 shrink-0 stroke-[1.8]" />
            <span className="truncate">{item.label}</span>
          </button>
        );
      })}

      <div className="my-1 h-px bg-outline/50" />

      <button
        type="button"
        onClick={() => setActiveSection("settings")}
        className={cn(
          "flex h-7 cursor-pointer items-center gap-2 rounded-md px-2 text-left text-body-sm font-medium transition-colors",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30",
          activeSection === "settings"
            ? "bg-surface-container-high text-on-surface"
            : "text-on-surface-variant hover:bg-control-hover hover:text-on-surface",
        )}
      >
        <Settings className="h-4 w-4 shrink-0 stroke-[1.8]" />
        <span className="truncate">Settings</span>
      </button>
    </nav>
  );
}
