import { Boxes, FileImage, Play, Plus } from "lucide-react";
import { Link, useLocation } from "@tanstack/react-router";
import { cn } from "@/lib/utils";
import * as m from "$paraglide/messages";

/* ───── Primary nav items ───── */

type PrimaryNavItem = {
  icon: React.ComponentType<{ className?: string }>;
  labelKey: string;
  to: string;
};

const PRIMARY_NAV_ITEMS: PrimaryNavItem[] = [
  { icon: Plus, labelKey: "sidebar.new", to: "/new" },
  { icon: Boxes, labelKey: "sidebar.models", to: "/models" },
  { icon: Play, labelKey: "sidebar.runs", to: "/runs" },
  { icon: FileImage, labelKey: "sidebar.assets", to: "/assets" },
];

export function SidebarNav({
  collapsed,
  children,
}: {
  collapsed?: boolean;
  children?: React.ReactNode;
}) {
  const { pathname } = useLocation();

  return (
    <nav
      className="flex flex-1 flex-col gap-0 overflow-y-auto px-2 pt-1"
      aria-label="Sidebar navigation"
    >
      <div className="space-y-0.5">
        {PRIMARY_NAV_ITEMS.map((item) => (
          <NavButton
            key={item.to}
            item={item}
            active={pathname === item.to}
            collapsed={collapsed}
          />
        ))}
      </div>
      {children}
    </nav>
  );
}

/* ───── Sub-components ───── */

function NavButton({
  item,
  active,
  collapsed,
}: {
  item: PrimaryNavItem;
  active: boolean;
  collapsed?: boolean;
}) {
  const Icon = item.icon;
  const label = (m as unknown as Record<string, () => string>)[item.labelKey]();

  return (
    <Link
      to={item.to}
      aria-current={active ? "page" : undefined}
      aria-label={collapsed ? label : undefined}
      title={collapsed ? label : undefined}
      className={cn(
        "flex h-8 w-full cursor-pointer items-center gap-2.5 rounded-md px-2.5 text-left text-sm transition-colors duration-150",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/20",
        collapsed && "justify-center px-0",
        active
          ? "bg-sidebar-item-active text-sidebar-text-primary"
          : "text-sidebar-text-secondary hover:bg-sidebar-item-hover hover:text-sidebar-text-primary",
      )}
    >
      <Icon
        className={cn(
          "h-4 w-4 shrink-0",
          active ? "text-sidebar-text-secondary" : "text-sidebar-text-muted",
        )}
      />
      {!collapsed && <span className="truncate">{label}</span>}
    </Link>
  );
}
