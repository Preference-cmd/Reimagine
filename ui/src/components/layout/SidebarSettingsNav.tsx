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
import { cn } from "@/lib/utils";
import { useUIStore } from "@/store/uiStore";
import * as m from "$paraglide/messages";
import { setLocale, getLocale } from "$paraglide/runtime";

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
      { id: "models", label: "", icon: Boxes },
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
 * SidebarSettingsNav — Codex-style light settings panel.
 * White background, grouped nav with icons, search, language switcher.
 */
export function SidebarSettingsNav() {
  const setActiveSidebarSection = useUIStore((s) => s.setActiveSidebarSection);
  const { active, setActive, search, setSearch, filtered, resolveLabel } = useSettingsNav();

  useEffect(() => {
    useUIStore.setState({ settingsNavId: active });
  }, [active]);

  return (
    <nav
      className="flex h-full flex-col border-r border-outline bg-surface"
      aria-label="Settings navigation"
    >
      {/* Back to app */}
      <div className="flex items-center gap-2 px-4 py-3">
        <button
          type="button"
          onClick={() => setActiveSidebarSection("chat")}
          className="flex items-center gap-1.5 text-sm text-on-surface-variant transition-colors hover:text-on-surface focus-visible:outline-none"
        >
          <ArrowLeft className="h-4 w-4" />
          {m["common.backToApp"]()}
        </button>
      </div>

      {/* Search */}
      <div className="px-3 pb-3">
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-on-surface-variant/50" />
          <input
            className="h-8 w-full rounded-lg border border-outline bg-surface-container-low px-3 pl-8 text-body-sm text-on-surface outline-none transition-[border-color,box-shadow] placeholder:text-on-surface-variant/50 focus-visible:border-secondary/30 focus-visible:ring-2 focus-visible:ring-secondary/10"
            placeholder={m["settings.searchPlaceholder"]()}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </div>

      {/* Language switcher */}
      <div className="px-3 pb-3">
        <div className="flex gap-1 rounded-lg bg-surface-container-low p-0.5">
          {(["en", "zh"] as const).map((code) => (
            <button
              key={code}
              type="button"
              onClick={() => setLocale(code)}
              className={cn(
                "flex-1 rounded-md px-3 py-1 text-xs font-medium transition-colors",
                getLocale() === code
                  ? "bg-surface text-on-surface shadow-sm"
                  : "text-on-surface-variant hover:text-on-surface hover:bg-control-hover",
              )}
            >
              {code === "en" ? "English" : "中文"}
            </button>
          ))}
        </div>
      </div>

      {/* Nav groups */}
      <div className="flex-1 overflow-y-auto scrollbar-hide px-2 pb-4">
        {filtered.map((group) => (
          <div key={group.section} className="mb-4">
            <div className="px-2 pb-1.5 text-[11px] font-semibold uppercase tracking-wider text-on-surface-variant/50">
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
                    "flex h-8 w-full cursor-pointer items-center gap-2.5 rounded-md px-2.5 text-left text-sm transition-colors",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-secondary/20",
                    isActive
                      ? "bg-secondary/10 text-secondary font-medium"
                      : "text-on-surface-variant hover:bg-control-hover hover:text-on-surface",
                  )}
                >
                  <Icon
                    className={cn(
                      "h-4 w-4 shrink-0",
                      isActive ? "text-secondary" : "text-on-surface-variant/60",
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
