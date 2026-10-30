import { Link, useLocation } from "@tanstack/react-router";
import { ArrowLeft, ArrowRight, Folder, PanelLeft, PanelRight } from "lucide-react";
import * as m from "$paraglide/messages";
import { cn } from "@/lib/utils";
import { ICON_STROKE } from "@/design/tokens";
import { useProjectStore } from "@/store/project";
import { useRuntimeStore, type RuntimePhase } from "@/store/runtime";
import { useSettingsStore } from "@/store/settingsStore";
import { useUIStore } from "@/store/uiStore";
import {
  isMacPlatform,
  isTauriApp,
  ShellIconButton,
  showsSessionPanel,
  type ShellHistory,
  WindowLights,
} from "./shellChrome";

function modeLabel(pathname: string): string {
  switch (pathname) {
    case "/":
      return m["shell.modeCanvas"]();
    case "/library":
      return m["shell.modeLibrary"]();
    case "/models":
      return m["shell.modeModules"]();
    case "/settings":
      return m["shell.modeSettings"]();
    default:
      return m["shell.modeCollection"]();
  }
}

function runPhaseLabel(phase: Exclude<RuntimePhase, "idle">): string {
  switch (phase) {
    case "starting":
      return m["shell.runPhaseStarting"]();
    case "running":
      return m["shell.runPhaseRunning"]();
    case "completed":
      return m["shell.runPhaseCompleted"]();
    case "failed":
      return m["shell.runPhaseFailed"]();
    case "cancelled":
      return m["shell.runPhaseCancelled"]();
  }
}

function runDotClass(phase: Exclude<RuntimePhase, "idle">): string {
  switch (phase) {
    case "starting":
    case "running":
      return "bg-info shadow-[0_0_0_3px_var(--color-info-tint)] animate-pulse";
    case "completed":
      return "bg-ok";
    case "failed":
      return "bg-err";
    case "cancelled":
      return "bg-fg4";
  }
}

/** 48px topbar: window controls when collapsed, breadcrumb, run pill, panel toggle. */
export function WorkbenchTopBar({ history }: { history: ShellHistory }) {
  const { pathname } = useLocation();
  const projects = useProjectStore((state) => state.projects);
  const activeProjectId = useProjectStore((state) => state.activeProjectId);
  const phase = useRuntimeStore((state) => state.phase);
  const runId = useRuntimeStore((state) => state.runId);
  const workflowName = useRuntimeStore((state) => state.workflowName);
  const progress = useRuntimeStore((state) => state.progress);
  const panelOpen = useUIStore((state) => state.sessionPanelOpen);
  const toggleSessionPanel = useUIStore((state) => state.toggleSessionPanel);
  const collapsed = useSettingsStore((state) => state.sidebarCollapsed);
  const setSidebarCollapsed = useSettingsStore((state) => state.setSidebarCollapsed);
  const { canGoBack, canGoForward, goBack, goForward } = history;

  const workspaceName =
    projects.find((project) => project.id === activeProjectId)?.name ?? activeProjectId;
  const tauri = isTauriApp();
  const mac = isMacPlatform();
  const sessionPanelVisible = showsSessionPanel(pathname);
  const runName = workflowName || runId;
  const runPercent = phase === "completed" ? 100 : Math.round(progress * 100);

  return (
    <header
      className={cn(
        "relative z-20 flex h-12 shrink-0 items-center gap-1.5 bg-surface px-3 pl-4 shadow-[0_1px_0_var(--color-hair-soft)]",
        collapsed && mac && tauri && "pl-[78px]",
      )}
    >
      <div data-tauri-drag-region aria-hidden="true" className="absolute inset-0" />

      <div className="relative z-10 flex min-w-0 flex-1 items-center gap-1.5">
        {collapsed && (
          <div className="mr-3 flex items-center gap-0.5">
            {mac && !tauri && <WindowLights />}
            <ShellIconButton
              label={m["shell.showSidebar"]()}
              onClick={() => setSidebarCollapsed(false)}
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
        )}

        <div className="flex min-w-0 items-center gap-2 text-ui text-fg3">
          <Folder className="size-3.5 shrink-0" strokeWidth={ICON_STROKE} />
          <span className="truncate">{workspaceName}</span>
          <span className="shrink-0 text-fg4">/</span>
          <b className="truncate font-medium tracking-[-0.005em] text-fg">{modeLabel(pathname)}</b>
        </div>
      </div>

      <div className="relative z-10 flex items-center gap-1.5">
        {phase !== "idle" && (
          <Link
            to="/library"
            search={{ type: "run" }}
            className="flex h-7 max-w-64 items-center gap-2 rounded-full bg-surface px-2.5 text-xs text-fg2 shadow-[0_0_0_1px_var(--color-hair),var(--shadow-1)] transition-[color,box-shadow,transform] duration-150 hover:text-fg hover:shadow-[0_0_0_1px_var(--color-hair-strong),var(--shadow-2)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            <span
              aria-hidden="true"
              className={cn("size-1.5 shrink-0 rounded-full", runDotClass(phase))}
            />
            <span className="shrink-0">{runPhaseLabel(phase)}</span>
            {runName && <span className="min-w-0 truncate text-fg3">{runName}</span>}
            <span className="shrink-0 font-mono tabular-nums text-fg3">{runPercent}%</span>
          </Link>
        )}

        {sessionPanelVisible && (
          <ShellIconButton
            label={m["shell.sessionPanel"]()}
            aria-pressed={panelOpen}
            className={cn(panelOpen && "bg-hover text-fg")}
            onClick={toggleSessionPanel}
          >
            <PanelRight className="size-4" strokeWidth={ICON_STROKE} />
          </ShellIconButton>
        )}
      </div>
    </header>
  );
}
