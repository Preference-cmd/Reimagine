import { useState, type ComponentType } from "react";
import { Link, useLocation, useNavigate } from "@tanstack/react-router";
import {
  ArrowLeft,
  ArrowRight,
  Blocks,
  Folder,
  LayoutGrid,
  Library,
  MoreHorizontal,
  Moon,
  PanelLeft,
  Play,
  Plus,
  Search,
  Settings,
  SquarePen,
  Sun,
} from "lucide-react";
import { toast } from "sonner";
import * as m from "$paraglide/messages";
import { getLocale, setLocale } from "$paraglide/runtime";
import { cn } from "@/lib/utils";
import { ICON_STROKE } from "@/design/tokens";
import { createProject } from "@/ipc";
import { useProjectStore } from "@/store/project";
import { useSettingsStore } from "@/store/settingsStore";
import { useUIStore } from "@/store/uiStore";
import { useTheme } from "@/lib/theme";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import {
  applyDocumentLang,
  formatRelativeTime,
  isMacPlatform,
  isTauriApp,
  SHELL_NAV_ITEM,
  ShellIconButton,
  type ShellHistory,
  WindowLights,
} from "./shellChrome";

function ModeLink({
  to,
  active,
  current = active,
  label,
  icon: Icon,
}: {
  to: "/" | "/library";
  active: boolean;
  current?: boolean;
  label: string;
  icon: ComponentType<{ className?: string; strokeWidth?: number }>;
}) {
  return (
    <Link
      to={to}
      activeOptions={{ exact: true }}
      aria-label={label}
      aria-current={current ? "page" : undefined}
      title={active ? undefined : label}
      className={cn(
        "relative flex h-8 items-center justify-center gap-1.5 rounded-md text-ui transition-colors duration-200",
        active
          ? "bg-accent-tint px-3 font-medium text-accent shadow-[inset_0_0_0_1px_var(--color-accent-line),var(--shadow-hi)]"
          : "px-2 text-fg3 hover:bg-hover hover:text-fg",
      )}
    >
      <Icon className="size-4 shrink-0" strokeWidth={ICON_STROKE} />
      {active && <span className="whitespace-nowrap">{label}</span>}
    </Link>
  );
}

/** Left rail: window controls, modes, workspaces, and the fixed bottom nav. */
export function WorkbenchSidebar({ history }: { history: ShellHistory }) {
  const { pathname, searchStr } = useLocation();
  const navigate = useNavigate();
  const projects = useProjectStore((state) => state.projects);
  const activeProjectId = useProjectStore((state) => state.activeProjectId);
  const switchProject = useProjectStore((state) => state.switchProject);
  const setSidebarCollapsed = useSettingsStore((state) => state.setSidebarCollapsed);
  const openCommandPalette = useUIStore((state) => state.openCommandPalette);
  const { themeMode, onThemeModeChange } = useTheme();
  const { canGoBack, canGoForward, goBack, goForward } = history;

  const [createOpen, setCreateOpen] = useState(false);
  const [name, setName] = useState("");
  const [creating, setCreating] = useState(false);

  const isCanvas = pathname === "/";
  const isLibrary = pathname === "/library";
  const runsFilter = isLibrary && new URLSearchParams(searchStr).get("type") === "run";
  const tauri = isTauriApp();
  const mac = isMacPlatform();

  const handleCreateOpenChange = (open: boolean) => {
    setCreateOpen(open);
    if (open) setName("");
  };

  const handleSwitchProject = async (projectId: string) => {
    try {
      await switchProject(projectId);
    } catch {
      toast.error(m["shell.switchWorkspaceFailed"]());
    }
  };

  const submitWorkspace = async (event: React.FormEvent) => {
    event.preventDefault();
    const trimmed = name.trim();
    if (!trimmed || creating) return;
    setCreating(true);
    try {
      const project = await createProject(`workspace-${crypto.randomUUID()}`, {
        name: trimmed,
        description: "",
      });
      await useProjectStore.getState().load();
      await switchProject(project.id);
      setCreateOpen(false);
      setName("");
    } catch {
      toast.error(m["shell.createWorkspaceFailed"]());
    } finally {
      setCreating(false);
    }
  };

  return (
    <aside
      aria-label={m["shell.sidebarLabel"]()}
      className="flex w-[var(--spacing-sidebar)] shrink-0 flex-col px-3 pb-3"
    >
      {/* 64px title row: shares a line with the 48px topbar inside the inset surface. */}
      <div className="relative flex h-16 shrink-0 items-center gap-0.5 pt-2">
        <div data-tauri-drag-region aria-hidden="true" className="absolute inset-0" />
        {mac && !tauri && <WindowLights className="relative z-10 mr-4" />}
        <div className={cn("relative z-10 flex items-center gap-0.5", mac && tauri && "pl-[78px]")}>
          <ShellIconButton
            label={m["shell.hideSidebar"]()}
            onClick={() => setSidebarCollapsed(true)}
          >
            <PanelLeft className="size-4" strokeWidth={ICON_STROKE} />
          </ShellIconButton>
          <ShellIconButton label={m["shell.back"]()} disabled={!canGoBack} onClick={goBack}>
            <ArrowLeft className="size-4" strokeWidth={ICON_STROKE} />
          </ShellIconButton>
          <ShellIconButton
            label={m["shell.forward"]()}
            disabled={!canGoForward}
            onClick={goForward}
          >
            <ArrowRight className="size-4" strokeWidth={ICON_STROKE} />
          </ShellIconButton>
        </div>
      </div>

      <div role="group" aria-label={m["shell.mainArea"]()} className="mb-3 flex gap-1">
        <ModeLink to="/" active={isCanvas} label={m["shell.modeCanvas"]()} icon={LayoutGrid} />
        <ModeLink
          to="/library"
          active={isLibrary}
          current={isLibrary && !runsFilter}
          label={m["shell.modeLibrary"]()}
          icon={Library}
        />
      </div>

      <span title={m["shell.agentNotConnected"]()} className="block">
        <button
          type="button"
          aria-disabled="true"
          aria-describedby="shell-new-session-hint"
          className={SHELL_NAV_ITEM}
        >
          <SquarePen className="size-4 shrink-0 text-fg2" strokeWidth={ICON_STROKE} />
          <span className="min-w-0 flex-1 truncate">{m["shell.newSession"]()}</span>
        </button>
        <span id="shell-new-session-hint" className="sr-only">
          {m["shell.agentNotConnected"]()}
        </span>
      </span>

      <button type="button" onClick={openCommandPalette} className={SHELL_NAV_ITEM}>
        <Search className="size-4 shrink-0 text-fg2" strokeWidth={ICON_STROKE} />
        <span className="min-w-0 flex-1 truncate">{m["shell.search"]()}</span>
      </button>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="group/section mt-[18px] flex h-7 items-center justify-between pl-2.5 pr-1">
          <span className="text-xs tracking-[0.02em] text-fg4">{m["shell.workspaces"]()}</span>
          <ShellIconButton
            label={m["shell.newWorkspace"]()}
            className="size-6 opacity-0 focus-visible:opacity-100 group-hover/section:opacity-100"
            onClick={() => handleCreateOpenChange(true)}
          >
            <Plus className="size-3.5" strokeWidth={ICON_STROKE} />
          </ShellIconButton>
        </div>

        {projects.length === 0 ? (
          <div className="px-2.5 py-2 text-xs leading-6 text-fg2">
            <p>{m["shell.noWorkspaces"]()}</p>
            <p>{m["shell.noWorkspacesHint"]()}</p>
          </div>
        ) : (
          <div className="mt-1 space-y-0.5">
            {projects.map((project) => {
              const active = project.id === activeProjectId;
              const updated = formatRelativeTime(project.updatedAt, getLocale());
              return (
                <button
                  key={project.id}
                  type="button"
                  aria-current={active ? "true" : undefined}
                  onClick={() => void handleSwitchProject(project.id)}
                  className={cn(
                    "flex h-8 w-full cursor-pointer items-center gap-2.5 rounded-sm px-2.5 text-left text-ui transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent",
                    active ? "bg-active text-fg2" : "text-fg2 hover:bg-hover hover:text-fg",
                  )}
                >
                  <Folder className="size-4 shrink-0" strokeWidth={ICON_STROKE} />
                  <span className="min-w-0 flex-1 truncate">{project.name}</span>
                  {updated && (
                    <span className="shrink-0 text-xs tabular-nums text-fg4">{updated}</span>
                  )}
                </button>
              );
            })}
          </div>
        )}
      </div>

      <div className="mt-2 shrink-0 pt-2 shadow-[0_-1px_0_var(--color-hair)]">
        <Link
          to="/models"
          aria-current={pathname === "/models" ? "page" : undefined}
          className={SHELL_NAV_ITEM}
        >
          <Blocks className="size-4 shrink-0 text-fg2" strokeWidth={ICON_STROKE} />
          <span className="min-w-0 flex-1 truncate">{m["shell.modules"]()}</span>
        </Link>
        <Link
          to="/library"
          search={{ type: "run" }}
          aria-current={runsFilter ? "page" : undefined}
          className={SHELL_NAV_ITEM}
        >
          <Play className="size-4 shrink-0 text-fg2" strokeWidth={ICON_STROKE} />
          <span className="min-w-0 flex-1 truncate">{m["shell.runs"]()}</span>
        </Link>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button type="button" className={SHELL_NAV_ITEM}>
              <MoreHorizontal className="size-4 shrink-0 text-fg2" strokeWidth={ICON_STROKE} />
              <span className="min-w-0 flex-1 truncate">{m["shell.more"]()}</span>
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent side="top" align="start" sideOffset={6} className="w-44">
            <DropdownMenuItem onSelect={() => void navigate({ to: "/settings" })}>
              <Settings className="size-4" strokeWidth={ICON_STROKE} />
              {m["shell.settings"]()}
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuLabel>{m["shell.language"]()}</DropdownMenuLabel>
            <DropdownMenuRadioGroup
              value={getLocale()}
              onValueChange={(value) => {
                const locale = value as "en" | "zh";
                setLocale(locale);
                applyDocumentLang(locale);
              }}
            >
              <DropdownMenuRadioItem value="en">
                {m["shell.languageEnglish"]()}
              </DropdownMenuRadioItem>
              <DropdownMenuRadioItem value="zh">
                {m["shell.languageChinese"]()}
              </DropdownMenuRadioItem>
            </DropdownMenuRadioGroup>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      <div className="mt-2 flex shrink-0 items-center gap-2 border-t border-hair pt-3 pl-1.5">
        <span className="grid size-7 shrink-0 place-items-center rounded-full bg-surface text-xs font-semibold shadow-[0_0_0_1px_var(--color-hair-strong),var(--shadow-1)]">
          R
        </span>
        <span className="min-w-0 flex-1 truncate text-ui font-medium text-fg">
          {m["shell.localUser"]()}
        </span>
        <ShellIconButton
          label={m["shell.toggleTheme"]()}
          aria-pressed={themeMode === "dark"}
          onClick={() => onThemeModeChange(themeMode === "dark" ? "light" : "dark")}
        >
          {themeMode === "dark" ? (
            <Sun className="size-4" strokeWidth={ICON_STROKE} />
          ) : (
            <Moon className="size-4" strokeWidth={ICON_STROKE} />
          )}
        </ShellIconButton>
        <ShellIconButton
          label={m["shell.settings"]()}
          onClick={() => void navigate({ to: "/settings" })}
        >
          <Settings className="size-4" strokeWidth={ICON_STROKE} />
        </ShellIconButton>
      </div>

      <Dialog open={createOpen} onOpenChange={handleCreateOpenChange}>
        <DialogContent className="sm:max-w-[360px]">
          <form onSubmit={(event) => void submitWorkspace(event)}>
            <DialogHeader>
              <DialogTitle>{m["shell.newWorkspace"]()}</DialogTitle>
              <DialogDescription>{m["shell.newWorkspaceDescription"]()}</DialogDescription>
            </DialogHeader>
            <div className="mt-4 space-y-1.5">
              <label htmlFor="shell-workspace-name" className="text-xs font-medium text-fg2">
                {m["shell.workspaceName"]()}
              </label>
              <Input
                id="shell-workspace-name"
                autoFocus
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder={m["shell.workspaceNamePlaceholder"]()}
                maxLength={120}
                className="text-ui"
              />
            </div>
            <DialogFooter className="mt-6">
              <Button type="button" variant="ghost" size="sm" onClick={() => setCreateOpen(false)}>
                {m["common.cancel"]()}
              </Button>
              <Button type="submit" size="sm" disabled={!name.trim() || creating}>
                {creating ? m["shell.creating"]() : m["shell.create"]()}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </aside>
  );
}
