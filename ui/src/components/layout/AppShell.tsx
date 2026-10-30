import { useEffect } from "react";
import { Outlet, useLocation } from "@tanstack/react-router";
import { ErrorBoundary } from "react-error-boundary";
import { Toaster } from "sonner";
import { getLocale } from "$paraglide/runtime";
import { cn } from "@/lib/utils";
import { useTheme } from "@/lib/theme";
import { useActivitySync } from "@/hooks/useActivitySync";
import { useProjectStore } from "@/store/project";
import { useSettingsStore } from "@/store/settingsStore";
import { CommandPalette } from "@/components/palette/CommandPalette";
import { ErrorFallback } from "@/components/layout/ErrorFallback";
import { SessionPanel } from "@/components/layout/SessionPanel";
import { WorkbenchSidebar } from "@/components/layout/WorkbenchSidebar";
import { WorkbenchTopBar } from "@/components/layout/WorkbenchTopBar";
import {
  applyDocumentLang,
  showsSessionPanel,
  useShellHistory,
} from "@/components/layout/shellChrome";

/**
 * Workbench shell: a fixed 264px rail and one inset surface holding the
 * 48px topbar, the routed area, and the floating session panel.
 */
export function AppShell() {
  const { themeMode } = useTheme();
  const sidebarCollapsed = useSettingsStore((state) => state.sidebarCollapsed);
  const history = useShellHistory();
  const { pathname } = useLocation();
  const sessionPanelVisible = showsSessionPanel(pathname);
  const activeProjectId = useProjectStore((state) => state.activeProjectId);

  // The feed is global: records arrive even while the panel is closed.
  useActivitySync(activeProjectId);

  // setLocale reloads the document, so the boot pass restores the lang the
  // switchers wrote before the reload.
  useEffect(() => {
    applyDocumentLang(getLocale() === "zh" ? "zh" : "en");
  }, []);

  return (
    <div className="flex h-full w-full overflow-hidden bg-app text-fg">
      {!sidebarCollapsed && <WorkbenchSidebar history={history} />}

      <main
        className={cn(
          "flex min-w-0 flex-1 flex-col pt-2 pr-2 pb-2",
          sidebarCollapsed ? "pl-2" : "pl-0",
        )}
      >
        <div className="relative flex min-h-0 flex-1 flex-col overflow-hidden rounded-[14px] bg-ground shadow-[0_0_0_1px_var(--color-hair),var(--shadow-2)]">
          <WorkbenchTopBar history={history} />
          <div className="relative flex min-h-0 flex-1 flex-col overflow-hidden">
            <ErrorBoundary FallbackComponent={ErrorFallback}>
              <Outlet />
            </ErrorBoundary>
            {sessionPanelVisible && <SessionPanel />}
          </div>
        </div>
      </main>

      <CommandPalette />
      <Toaster
        theme={themeMode}
        position="bottom-center"
        toastOptions={{
          className: "!font-sans",
          classNames: {
            toast: "!rounded-full !border !border-hair !bg-surface !text-fg !shadow-3",
            title: "!text-ui !font-medium",
            description: "!text-caption !text-fg2",
          },
        }}
      />
    </div>
  );
}
