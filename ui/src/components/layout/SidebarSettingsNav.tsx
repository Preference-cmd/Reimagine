import {
  ArrowLeft,
  Boxes,
  Globe,
  Keyboard,
  Layers,
  Monitor,
  Palette,
  Search,
  Settings,
  SlidersHorizontal,
  type LucideIcon,
} from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { cn } from "@/lib/utils";
import { useUIStore } from "@/store/uiStore";
import * as m from "$paraglide/messages";

type NavItem = { id: string; label: string; icon: LucideIcon };
type NavGroup = { section: string; sectionKey: string; items: NavItem[] };

const SETTINGS_NAV: NavGroup[] = [
  {
    section: "Personal",
    sectionKey: "settings.groupPersonal",
    items: [
      { id: "general", label: "", icon: Settings },
      { id: "appearance", label: "", icon: Palette },
      { id: "shortcuts", label: "", icon: Keyboard },
      { id: "runtime", label: "", icon: Monitor },
      { id: "workspace", label: "", icon: Layers },
      { id: "modelManagement", label: "", icon: Boxes },
    ],
  },
  {
    section: "Integration",
    sectionKey: "settings.groupIntegration",
    items: [
      { id: "plugins", label: "", icon: Boxes },
      { id: "browser", label: "", icon: Globe },
    ],
  },
  {
    section: "Coding",
    sectionKey: "settings.groupCoding",
    items: [{ id: "hooks", label: "", icon: SlidersHorizontal }],
  },
];

export type SettingsNavId = string;

export function useSettingsNav() {
  const [active, setActive] = useState<SettingsNavId>("general");
  const [search, setSearch] = useState("");

  const resolveLabel = (key: string): string => {
    const msg = (m as unknown as Record<string, () => string>)[key];
    return msg ? msg() : key;
  };

  const filtered = SETTINGS_NAV.map((group) => ({
    ...group,
    items: group.items
      .map((item) => ({
        ...item,
        label: resolveLabel(`settings.${item.id}`) || item.id,
      }))
      .filter((item) => !search || item.label.toLowerCase().includes(search.toLowerCase())),
  })).filter((group) => group.items.length > 0);

  return { active, setActive, search, setSearch, filtered, resolveLabel };
}

/**
 * SidebarSettingsNav — settings panel with sidebar-consistent theming.
 * Uses sidebar background and text colors for visual coherence.
 */
export function SidebarSettingsNav() {
  const navigate = useNavigate();
  const { active, setActive, search, setSearch, filtered, resolveLabel } = useSettingsNav();

  useEffect(() => {
    useUIStore.setState({ settingsNavId: active });
  }, [active]);

  return (
    <nav className="flex h-full flex-col bg-sidebar-bg" aria-label="Settings navigation">
      {/* Back to app */}
      <div className="flex items-center gap-2 px-4 py-3">
        <button
          type="button"
          onClick={() => navigate({ to: "/new" })}
          className="flex items-center gap-1.5 text-sm text-sidebar-text-secondary transition-colors hover:text-sidebar-text-primary focus-visible:outline-none"
        >
          <ArrowLeft className="h-4 w-4" />
          {m["common.backToApp"]()}
        </button>
      </div>

      {/* Search */}
      <div className="px-3 pb-3">
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-sidebar-text-muted" />
          <input
            className="h-8 w-full rounded-lg border border-sidebar-border bg-sidebar-item-hover px-3 pl-8 text-body-sm text-sidebar-text-primary outline-none transition-[border-color,box-shadow] placeholder:text-sidebar-text-muted focus-visible:border-sidebar-text-secondary/30 focus-visible:ring-2 focus-visible:ring-sidebar-text-secondary/10"
            placeholder={m["settings.searchPlaceholder"]()}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </div>

      {/* Nav groups */}
      <div className="flex-1 overflow-y-auto scrollbar-hide px-2 pb-4">
        {filtered.map((group) => (
          <div key={group.section} className="mb-5">
            <div className="px-3 pb-1.5 pt-2 text-[11px] font-semibold uppercase tracking-wider text-sidebar-section-header">
              {resolveLabel(group.sectionKey)}
            </div>
            {group.items.map((item) => {
              const isActive = item.id === active;
              const Icon = item.icon;
              return (
                <button
                  key={item.id}
                  type="button"
                  aria-current={isActive ? "page" : undefined}
                  onClick={() => setActive(item.id)}
                  className={cn(
                    "flex h-9 w-full cursor-pointer items-center gap-2.5 rounded-md px-3 text-left text-sm transition-colors",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-text-secondary/20",
                    isActive
                      ? "bg-sidebar-item-active text-sidebar-text-primary font-medium"
                      : "text-sidebar-text-secondary hover:bg-sidebar-item-hover hover:text-sidebar-text-primary",
                  )}
                >
                  <Icon
                    className={cn(
                      "h-4 w-4 shrink-0",
                      isActive ? "text-sidebar-text-primary" : "text-sidebar-text-muted",
                    )}
                  />
                  <span className="truncate">{item.label}</span>
                </button>
              );
            })}
          </div>
        ))}
      </div>
    </nav>
  );
}
