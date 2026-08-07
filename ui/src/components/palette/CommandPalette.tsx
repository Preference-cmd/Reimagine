import { useEffect, useMemo, useRef, useState } from "react";
import { Boxes, CircleDot, CornerDownLeft, Plus, Search } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { useUIStore } from "@/store/uiStore";
import { useNodeRegistryStore } from "@/store/nodeRegistry";
import { useRuntimeStore } from "@/store/runtime";
import { saveWorkflowNow } from "@/hooks/useWorkflowPersistence";
import { createNodeAt } from "@/lib/nodeFactory";
import { filterEntries, type PaletteEntry } from "@/lib/palette";
import { flowViewportCenter } from "@/lib/flowInstance";

/**
 * Command palette (F3-3): Cmd/Ctrl+P toggles a searchable action list;
 * "Add Node…" switches into a node-type picker that inserts at the viewport
 * center (or the canvas double-click position, F2-2). Fuzzy matching and
 * arrow-key navigation with a simple filter input — no extra deps.
 */
export function CommandPalette() {
  const palette = useUIStore((s) => s.palette);
  const openCommandPalette = useUIStore((s) => s.openCommandPalette);
  const openNodePalette = useUIStore((s) => s.openNodePalette);
  const closePalette = useUIStore((s) => s.closePalette);
  const startRun = useRuntimeStore((s) => s.startRun);
  const defs = useNodeRegistryStore((s) => s.defList);

  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

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

  const commandItems: PaletteEntry[] = useMemo(
    () => [
      {
        id: "add-node",
        label: "Add Node…",
        hint: "Insert a node from the catalog",
        keywords: ["create", "insert", "node"],
        shortcut: "⇧⌘N",
      },
      {
        id: "toggle-panel",
        label: "Toggle Sidebar",
        hint: "Show or hide the sidebar",
        keywords: ["explorer", "sidebar", "panel"],
        shortcut: "⌘B",
      },
      {
        id: "run-workflow",
        label: "Run Workflow",
        hint: "Start generation",
        keywords: ["run", "execute", "generate", "start"],
        shortcut: "⌘⏎",
      },
      {
        id: "save",
        label: "Save",
        hint: "Save workflow to workspace",
        keywords: ["persist", "write", "file"],
        shortcut: "⌘S",
      },
      {
        id: "settings",
        label: "Settings",
        hint: "Application settings",
        keywords: ["preferences", "config"],
        shortcut: "⌘,",
      },
    ],
    [],
  );

  const nodeItems: PaletteEntry[] = useMemo(
    () =>
      defs.map((def) => ({
        id: `node-${def.type}`,
        label: def.displayName,
        hint: def.type,
        keywords: [def.type, def.category],
      })),
    [defs],
  );

  const entries = palette.kind === "node" ? nodeItems : commandItems;
  const visible = filterEntries(query, entries);
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
      case "add-node":
        openNodePalette(flowViewportCenter());
        return;
      case "toggle-panel":
        useUIStore.getState().toggleSidebar();
        return;
      case "run-workflow":
        startRun();
        return;
      case "save":
        void saveWorkflowNow();
        return;
      case "settings":
        useUIStore.getState().setActiveSidebarSection("settings");
        return;
      default:
        if (id.startsWith("node-")) {
          const typeId = id.slice("node-".length);
          const position = palette.kind === "node" ? palette.position : flowViewportCenter();
          if (!createNodeAt(typeId, position)) {
            toast.error("Unknown node type", {
              description: `"${typeId}" is not in the node catalog.`,
            });
          }
        }
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
    } else if (event.key === "Escape") {
      event.preventDefault();
      closePalette();
    }
  };

  const backToCommands = () => {
    openCommandPalette();
  };

  return (
    <div
      className="pointer-events-auto fixed inset-0 z-[var(--overlay-z-modal)] flex items-start justify-center bg-black/20 pt-[18vh]"
      onPointerDown={(event) => {
        if (event.target === event.currentTarget) closePalette();
      }}
    >
      <div
        role="dialog"
        aria-label="Command palette"
        className="panel-raised w-[min(440px,calc(100vw-2rem))] overflow-hidden rounded-2xl"
      >
        <div className="flex items-center gap-2 border-b border-outline px-3.5 py-3">
          {palette.kind === "node" ? (
            <Plus className="h-4 w-4 shrink-0 text-on-surface-variant" />
          ) : (
            <Search className="h-4 w-4 shrink-0 text-on-surface-variant" />
          )}
          <input
            ref={inputRef}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={onKeyDown}
            placeholder={palette.kind === "node" ? "Search node types…" : "Search commands…"}
            className="min-w-0 flex-1 bg-transparent text-body-sm text-on-surface placeholder:text-on-surface-variant focus:outline-none"
          />
          {palette.kind === "node" && (
            <button
              type="button"
              onClick={backToCommands}
              className="shrink-0 rounded-md bg-control-hover px-2 py-1 text-caption font-medium text-on-surface-variant transition-colors hover:text-on-surface"
            >
              ← Commands
            </button>
          )}
        </div>

        <ul ref={listRef} className="scrollbar-hide max-h-80 overflow-y-auto p-1.5">
          {visible.length === 0 && (
            <li className="px-2.5 py-2 text-caption text-on-surface-variant">No matches.</li>
          )}
          {visible.map((entry, index) => {
            const Icon =
              palette.kind === "node" ? Boxes : entry.id === "add-node" ? Plus : CircleDot;
            return (
              <li key={entry.id}>
                <button
                  type="button"
                  data-active={index === activeIndex}
                  onMouseEnter={() => setActiveIndex(index)}
                  onClick={() => runCommand(entry.id)}
                  className={cn(
                    "flex w-full cursor-pointer items-center gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30",
                    index === activeIndex && "bg-control-hover",
                  )}
                >
                  <span
                    className={cn(
                      "flex h-6 w-6 shrink-0 items-center justify-center rounded-md",
                      index === activeIndex
                        ? "bg-primary text-on-primary"
                        : "bg-control-hover text-on-surface-variant",
                    )}
                  >
                    <Icon className="h-3.5 w-3.5" />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-body-sm font-medium text-on-surface">
                      {entry.label}
                    </span>
                    {entry.hint && (
                      <span className="block truncate text-caption text-on-surface-variant">
                        {entry.hint}
                      </span>
                    )}
                  </span>
                  {entry.shortcut && (
                    <span className="flex shrink-0 items-center gap-1 text-caption text-on-surface-variant/70">
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
