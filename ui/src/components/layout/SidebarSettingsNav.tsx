import { ArrowLeft, Search } from "lucide-react";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { useUIStore } from "@/store/uiStore";
import * as m from "$paraglide/messages";
import { setLocale, getLocale } from "$paraglide/runtime";

type NavItem = { id: string; label: string; icon: string };
type NavGroup = { section: string; items: NavItem[] };

const SETTINGS_NAV: NavGroup[] = [
  {
    section: m["settings.personal"](),
    items: [
      { id: "general", label: m["settings.general"](), icon: "⚙" },
      { id: "appearance", label: m["settings.appearance"](), icon: "🎨" },
      { id: "shortcuts", label: m["settings.shortcuts"](), icon: "⌨" },
      { id: "runtime", label: m["settings.runtime"](), icon: "🖥" },
    ],
  },
  {
    section: m["settings.project"](),
    items: [
      { id: "workspace", label: m["settings.workspace"](), icon: "📁" },
      { id: "models", label: m["settings.modelManagement"](), icon: "📦" },
    ],
  },
];

export type SettingsNavId = string;

export function useSettingsNav() {
  const [active, setActive] = useState<SettingsNavId>("general");
  const [search, setSearch] = useState("");

  const filtered = SETTINGS_NAV.map((group) => ({
    ...group,
    items: group.items.filter(
      (item) => !search || item.label.toLowerCase().includes(search.toLowerCase()),
    ),
  })).filter((group) => group.items.length > 0);

  return { active, setActive, search, setSearch, filtered };
}

/**
 * SidebarSettingsNav — shown when activeSidebarSection === "settings".
 * Codex-style dark theme settings panel.
 */
export function SidebarSettingsNav() {
  const setActiveSidebarSection = useUIStore((s) => s.setActiveSidebarSection);
  const { active, setActive, search, setSearch, filtered } = useSettingsNav();

  useEffect(() => {
    useUIStore.setState({ settingsNavId: active });
  }, [active]);

  return (
    <nav className="flex h-full flex-col bg-[#0d0d0d]" aria-label="Settings navigation">
      {/* Back to app */}
      <button
        type="button"
        onClick={() => setActiveSidebarSection("workflows")}
        className="flex h-10 w-full cursor-pointer items-center gap-2 px-3 text-sm font-medium text-white/50 transition-colors hover:bg-white/[0.06] hover:text-white/80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/20"
      >
        <ArrowLeft className="h-4 w-4" />
        {m["common.backToApp"]()}
      </button>

      {/* Language switcher */}
      <div className="px-3 pb-3">
        <div className="flex gap-1 rounded-lg bg-white/[0.04] p-0.5">
          {(["en", "zh"] as const).map((code) => (
            <button
              key={code}
              type="button"
              onClick={() => setLocale(code)}
              className={cn(
                "flex-1 rounded-md px-3 py-1 text-xs font-medium transition-colors",
                getLocale() === code
                  ? "bg-white/[0.08] text-white"
                  : "text-white/40 hover:text-white/70",
              )}
            >
              {code === "en" ? "English" : "中文"}
            </button>
          ))}
        </div>
      </div>

      {/* Search */}
      <div className="px-3 pb-3">
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-white/30" />
          <input
            className="h-7 w-full rounded-lg border border-white/[0.06] bg-white/[0.04] px-3 pl-8 text-xs text-white/80 outline-none transition-[border-color,box-shadow] placeholder:text-white/25 focus-visible:border-white/[0.12] focus-visible:ring-1 focus-visible:ring-white/[0.08]"
            placeholder={m["settings.searchPlaceholder"]()}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </div>

      {/* Nav groups */}
      <div className="flex-1 overflow-y-auto scrollbar-hide px-2 pb-4">
        {filtered.map((group) => (
          <div key={group.section} className="mb-3">
            <div className="px-2 pb-1 text-[10px] font-semibold uppercase tracking-wider text-white/20">
              {group.section}
            </div>
            {group.items.map((item) => {
              const isActive = item.id === active;
              return (
                <button
                  key={item.id}
                  type="button"
                  aria-current={isActive ? "page" : undefined}
                  onClick={() => setActive(item.id)}
                  className={cn(
                    "flex h-7 w-full cursor-pointer items-center gap-2 rounded-md px-2 text-left text-sm transition-colors",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/20",
                    isActive
                      ? "bg-white/[0.08] text-white"
                      : "text-white/40 hover:bg-white/[0.06] hover:text-white/70",
                  )}
                >
                  <span className="text-xs">{item.icon}</span>
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
