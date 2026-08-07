import { Workflow, Boxes, History, Image } from "lucide-react";
import { cn } from "@/lib/utils";
import { useUIStore, type SidebarSection } from "@/store/uiStore";
import * as m from "$paraglide/messages";

type NavItem = {
  icon: React.ComponentType<{ className?: string }>;
  labelKey: string;
  section: SidebarSection;
};

const NAV_ITEMS: NavItem[] = [
  { icon: Workflow, labelKey: "sidebar.workflows", section: "workflows" },
  { icon: Boxes, labelKey: "sidebar.models", section: "models" },
  { icon: History, labelKey: "sidebar.runs", section: "runs" },
  { icon: Image, labelKey: "sidebar.assets", section: "assets" },
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
  // Cast through unknown to allow dynamic key lookup on Paraglide messages
  const label = (m as unknown as Record<string, () => string>)[item.labelKey]();

  return (
    <button
      type="button"
      aria-current={active ? "page" : undefined}
      onClick={onClick}
      className={cn(
        "flex h-8 w-full cursor-pointer items-center gap-2.5 rounded-md px-2.5 text-left text-sm transition-colors duration-150",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/20",
        active
          ? "bg-white/[0.08] text-white"
          : "text-white/40 hover:bg-white/[0.05] hover:text-white/70",
      )}
    >
      <Icon className={cn("h-4 w-4 shrink-0", active ? "text-white/70" : "text-white/30")} />
      <span className="truncate">{label}</span>
    </button>
  );
}

export function SidebarNav() {
  const activeSection = useUIStore((s) => s.activeSidebarSection);
  const setActiveSection = useUIStore((s) => s.setActiveSidebarSection);

  return (
    <nav className="flex flex-1 flex-col gap-0.5 px-2 pt-1" aria-label="Sidebar navigation">
      {NAV_ITEMS.map((item) => (
        <NavButton
          key={item.section}
          item={item}
          active={activeSection === item.section}
          onClick={() => setActiveSection(item.section)}
        />
      ))}
    </nav>
  );
}
