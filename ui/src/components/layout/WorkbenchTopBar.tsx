import { useLocation } from "@tanstack/react-router";
import { ArrowLeft, ArrowRight, Folder, PanelLeft, PanelRight } from "lucide-react";
import * as m from "$paraglide/messages";
import { cn } from "@/lib/utils";
import { ICON_STROKE } from "@/design/tokens";
import { useProjectStore } from "@/store/project";
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

/** 48px topbar: window controls when collapsed, breadcrumb, panel toggle. */
export function WorkbenchTopBar({ history }: { history: ShellHistory }) {
  const { pathname } = useLocation();
  const projects = useProjectStore((state) => state.projects);
  const activeProjectId = useProjectStore((state) => state.activeProjectId);
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

  return (
    <header
      data-tauri-drag-region
      className={cn(
        "relative z-20 flex h-12 shrink-0 items-center gap-1.5 bg-surface px-3 pl-4 shadow-[0_1px_0_var(--color-hair-soft)]",
        collapsed && mac && tauri && "pl-[78px]",
      )}
    >
      <div className="flex min-w-0 flex-1 items-center gap-1.5">
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

        <div
          data-tauri-drag-region
          className="flex min-w-0 flex-1 select-none items-center gap-2 text-ui text-fg2"
        >
          <Folder className="pointer-events-none size-3.5 shrink-0" strokeWidth={ICON_STROKE} />
          <span className="pointer-events-none truncate">{workspaceName}</span>
          <span className="pointer-events-none shrink-0 text-fg4">/</span>
          <b className="pointer-events-none truncate font-medium tracking-[-0.005em] text-fg">
            {modeLabel(pathname)}
          </b>
        </div>
      </div>

      <div className="flex items-center gap-1.5">
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
