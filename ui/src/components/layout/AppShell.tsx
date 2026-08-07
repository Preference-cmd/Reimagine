import { useEffect, useState } from "react";
import { ErrorBoundary } from "react-error-boundary";
import { Toaster } from "sonner";
import { Sidebar } from "./Sidebar";
import { WindowControls } from "./WindowControls";
import { PropertiesDrawer } from "./PropertiesDrawer";
import { RunFab } from "./RunFab";
import { MainContent } from "./MainContent";
import { ErrorFallback } from "./ErrorFallback";
import { ContextMenuPanel } from "@/components/canvas/ContextMenuPanel";
import { RenameNodeDialog } from "@/components/canvas/RenameNodeDialog";
import { CommandPalette } from "@/components/palette/CommandPalette";
import { useNodeRegistryStore } from "@/store/nodeRegistry";
import { getRoute } from "@/lib/routes";
import { useUIStore } from "@/store/uiStore";
import type { ThemeMode } from "./SettingsView";

const THEME_STORAGE_KEY = "reimagine.theme";

function readStoredTheme(): ThemeMode {
  if (typeof window === "undefined") {
    return "light";
  }
  const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
  return stored === "dark" ? "dark" : "light";
}

/**
 * AppShell — root layout for the editor workspace.
 *
 * Codex-style sidebar-first layout:
 *   - Sidebar (56px icon bar) on the left, expands to 260px for settings
 *   - Main content area switches between views based on sidebar nav
 *   - PropertiesDrawer slides from the right edge
 *   - Global overlays (CommandPalette, ContextMenu, RenameDialog, Toaster)
 */
export function AppShell() {
  const activeSection = useUIStore((s) => s.activeSidebarSection);
  const [themeMode, setThemeMode] = useState<ThemeMode>(readStoredTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = themeMode;
    document.documentElement.classList.toggle("dark", themeMode === "dark");
    window.localStorage.setItem(THEME_STORAGE_KEY, themeMode);
  }, [themeMode]);

  // Fetch the backend node catalog once at startup.
  useEffect(() => {
    void useNodeRegistryStore.getState().load();
  }, []);

  const handleThemeModeChange = (mode: ThemeMode) => {
    setThemeMode(mode);
    window.localStorage.setItem(THEME_STORAGE_KEY, mode);
  };

  // Use route config for conditional UI elements
  const route = getRoute(activeSection);

  return (
    <div className="app-shell flex h-full w-full bg-background text-foreground">
      {/* Sidebar — fixed left column */}
      <Sidebar />

      {/* Main content — fills remaining space */}
      <main className="main-content relative flex min-w-0 flex-1 flex-col overflow-hidden">
        <WindowControls />

        <div className="relative min-h-0 flex-1">
          <ErrorBoundary FallbackComponent={ErrorFallback}>
            <MainContent themeMode={themeMode} onThemeModeChange={handleThemeModeChange} />
          </ErrorBoundary>
          {route.showRunFab && <RunFab />}
        </div>
      </main>

      {/* Properties drawer — slides from right */}
      {route.showPropertiesDrawer && <PropertiesDrawer />}

      {/* Global overlays */}
      <CommandPalette />
      <ContextMenuPanel />
      <RenameNodeDialog />
      <Toaster
        theme={themeMode}
        position="bottom-center"
        toastOptions={{
          className: "!font-sans",
          classNames: {
            toast: "!rounded-xl !border-outline !bg-surface-container-high !text-on-surface",
            title: "!text-body-sm !font-medium",
            description: "!text-caption !text-on-surface-variant",
          },
        }}
      />
    </div>
  );
}
