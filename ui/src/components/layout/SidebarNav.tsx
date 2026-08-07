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

function NavButton({
  item,
  active,
  onClick,
}: {
  item: NavItem;
  active: boolean;
  onClick: () => void;
}) {
  const Icon = item.icon;
  return (
    <button
      type="button"
      aria-current={active ? "page" : undefined}
      onClick={onClick}
      className={cn(
        "group relative flex h-8 cursor-pointer items-center gap-2.5 rounded-lg px-2.5 text-left text-body-sm font-medium transition-all duration-150",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30",
        active
          ? "bg-surface-container-high text-on-surface"
          : "text-on-surface-variant hover:bg-control-hover hover:text-on-surface",
      )}
    >
      {/* Active indicator bar */}
      {active && (
        <span className="absolute -left-0.5 top-1/2 h-4 w-0.5 -translate-y-1/2 rounded-full bg-primary" />
      )}
      <Icon
        className={cn(
          "h-4 w-4 shrink-0 stroke-[1.8]",
          active ? "text-on-surface" : "text-on-surface-variant/70 group-hover:text-on-surface",
        )}
      />
      <span className="truncate">{item.label}</span>
    </button>
  );
}

export function SidebarNav() {
  const activeSection = useUIStore((s) => s.activeSidebarSection);
  const setActiveSection = useUIStore((s) => s.setActiveSidebarSection);

  return (
    <nav className="flex flex-col gap-1 px-3 pt-1" aria-label="Sidebar navigation">
      {NAV_ITEMS.map((item) => (
        <NavButton
          key={item.section}
          item={item}
          active={activeSection === item.section}
          onClick={() => setActiveSection(item.section)}
        />
      ))}

      <div className="my-2 h-px bg-outline/40" />

      <NavButton
        item={{ icon: Settings, label: "Settings", section: "settings" }}
        active={activeSection === "settings"}
        onClick={() => setActiveSection("settings")}
      />
    </nav>
  );
}
