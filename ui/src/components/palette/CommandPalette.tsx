import { useEffect, useRef, useState } from "react";
import {
  CircleDot,
  CornerDownLeft,
  LayoutGrid,
  Library,
  PanelLeft,
  Search,
  Settings,
} from "lucide-react";
import { useNavigate } from "@tanstack/react-router";
import * as m from "$paraglide/messages";
import { cn } from "@/lib/utils";
import { useUIStore } from "@/store/uiStore";
import { useSettingsStore } from "@/store/settingsStore";
import { filterEntries, type PaletteEntry } from "@/lib/palette";

const COMMAND_ICONS: Record<string, typeof CircleDot> = {
  "open-canvas": LayoutGrid,
  "open-library": Library,
  "toggle-panel": PanelLeft,
  settings: Settings,
};

/**
 * Command palette (F3-3): Cmd/Ctrl+P toggles a searchable action list.
 * Fuzzy matching and arrow-key navigation with a simple filter input —
 * no extra deps.
 */
export function CommandPalette() {
  const palette = useUIStore((s) => s.palette);
  const openCommandPalette = useUIStore((s) => s.openCommandPalette);
  const closePalette = useUIStore((s) => s.closePalette);
  const navigate = useNavigate();

  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);

  const open = palette.kind !== "closed";

  useEffect(() => {
    if (!open) {
      setQuery("");
      setActiveIndex(0);
      return;
    }
    setActiveIndex(0);
    inputRef.current?.focus();
  }, [open, palette.kind]);

  /* Escape closes regardless of which control inside the palette has focus. */
  useEffect(() => {
    if (!open) return;
    const handler = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      closePalette();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open, closePalette]);

  /* Cmd/Ctrl+P toggle — kept self-contained (F3-2 not built yet). */
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const mod = event.metaKey || event.ctrlKey;
      if (!mod || event.key.toLowerCase() !== "p") return;
      event.preventDefault();
      openCommandPalette();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [openCommandPalette]);

  // Rebuilt every render so a locale switch reaches the labels without remount.
  const commandItems: PaletteEntry[] = [
    {
      id: "open-canvas",
      label: m["palette.openCanvas"](),
      hint: m["palette.openCanvasHint"](),
      keywords: ["canvas", "board", "project", "work"],
    },
    {
      id: "open-library",
      label: m["palette.openLibrary"](),
      hint: m["palette.openLibraryHint"](),
      keywords: ["library", "collections", "runs", "files", "assets"],
    },
    {
      id: "toggle-panel",
      label: m["palette.toggleSidebar"](),
      hint: m["palette.toggleSidebarHint"](),
      keywords: ["explorer", "sidebar", "panel"],
    },
    {
      id: "settings",
      label: m["palette.settings"](),
      hint: m["palette.settingsHint"](),
      keywords: ["preferences", "config"],
    },
  ];

  const visible = filterEntries(query, commandItems);
  const itemCount = visible.length;

  useEffect(() => {
    setActiveIndex(0);
  }, [query, palette.kind]);

  useEffect(() => {
    const active = listRef.current?.querySelector('[data-active="true"]');
    active?.scrollIntoView({ block: "nearest" });
  }, [activeIndex]);

  if (!open) return null;

  const runCommand = (id: string) => {
    closePalette();
    switch (id) {
      case "open-canvas":
        navigate({ to: "/" });
        return;
      case "open-library":
        navigate({ to: "/library" });
        return;
      case "toggle-panel": {
        const settings = useSettingsStore.getState();
        settings.setSidebarCollapsed(!settings.sidebarCollapsed);
        return;
      }
      case "settings":
        navigate({ to: "/settings" });
        return;
    }
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveIndex((index) => (itemCount === 0 ? 0 : (index + 1) % itemCount));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveIndex((index) => (itemCount === 0 ? 0 : (index - 1 + itemCount) % itemCount));
    } else if (event.key === "Enter") {
      event.preventDefault();
      const item = visible[activeIndex];
      if (item) runCommand(item.id);
    }
  };

  const trapFocus = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "Tab") return;
    const focusable = dialogRef.current?.querySelectorAll<HTMLElement>(
      'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    );
    if (!focusable || focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement;
    if (event.shiftKey && (active === first || !dialogRef.current?.contains(active))) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && active === last) {
      event.preventDefault();
      first.focus();
    }
  };

  return (
    <div
      className="pointer-events-auto fixed inset-0 z-50 flex items-start justify-center bg-[var(--color-scrim)] pt-[18vh]"
      onPointerDown={(event) => {
        if (event.target === event.currentTarget) closePalette();
      }}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label={m["palette.commandPalette"]()}
        onKeyDown={trapFocus}
        className="w-[min(440px,calc(100vw-2rem))] overflow-hidden rounded-2xl bg-surface shadow-[var(--shadow-bezel),var(--shadow-3),var(--shadow-hi)]"
      >
        <div className="flex items-center gap-2 border-b border-hair px-3.5 py-3">
          <Search className="h-4 w-4 shrink-0 text-fg3" />
          <input
            ref={inputRef}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={onKeyDown}
            placeholder={m["palette.searchCommands"]()}
            className="min-w-0 flex-1 bg-transparent text-body-sm text-fg placeholder:text-fg2 focus:outline-none"
          />
        </div>

        <ul ref={listRef} className="scrollbar-hide max-h-80 overflow-y-auto p-1.5">
          {visible.length === 0 && (
            <li className="px-2.5 py-2 text-caption text-fg2">{m["palette.noMatches"]()}</li>
          )}
          {visible.map((entry, index) => {
            const Icon = COMMAND_ICONS[entry.id] ?? CircleDot;
            return (
              <li key={entry.id}>
                <button
                  type="button"
                  data-active={index === activeIndex}
                  onMouseEnter={() => setActiveIndex(index)}
                  onClick={() => runCommand(entry.id)}
                  className={cn(
                    "flex w-full cursor-pointer items-center gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/30",
                    index === activeIndex && "bg-hover",
                  )}
                >
                  <span
                    className={cn(
                      "flex h-6 w-6 shrink-0 items-center justify-center rounded-md",
                      index === activeIndex ? "bg-accent text-on-accent" : "bg-hover text-fg2",
                    )}
                  >
                    <Icon className="h-3.5 w-3.5" />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-body-sm font-medium text-fg">
                      {entry.label}
                    </span>
                    {entry.hint && (
                      <span className="block truncate text-caption text-fg2">{entry.hint}</span>
                    )}
                  </span>
                  {entry.shortcut && (
                    <span className="flex shrink-0 items-center gap-1 text-caption text-fg4">
                      <CornerDownLeft className="h-3 w-3" />
                      {entry.shortcut}
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
