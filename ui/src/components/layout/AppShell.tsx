import { useEffect, useState } from "react";
import { TopBar } from "./TopBar";
import { SideRail } from "./SideRail";
import { ExplorerPanel } from "./ExplorerPanel";
import { PropertiesPanel } from "./PropertiesPanel";
import { SettingsPanel, type ThemeMode } from "./SettingsPanel";
import {
  getOverlayPolicy,
} from "./overlayLayout";
import { NodeCanvas } from "@/components/canvas/NodeCanvas";
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
 * Structure:
 *   - NodeCanvas lives inside an overflow-hidden layer so the canvas grid
 *     cannot scroll the viewport.
 *   - TopBar, SideRail, ExplorerPanel, and PropertiesPanel are siblings
 *     outside that clipping layer so tooltips/menus/popovers can escape.
 */
export function AppShell() {
  const [activePanel, setActivePanel] = useState<string | null>(null);
  const [themeMode, setThemeMode] = useState<ThemeMode>(readStoredTheme);
  const overlayPolicy = getOverlayPolicy(activePanel);

  useEffect(() => {
    document.documentElement.dataset.theme = themeMode;
    document.documentElement.classList.toggle("dark", themeMode === "dark");
    window.localStorage.setItem(THEME_STORAGE_KEY, themeMode);
  }, [themeMode]);

  return (
    <div className="overlay-root relative h-full w-full bg-background text-foreground">
      <div className="absolute inset-0">
        <NodeCanvas themeMode={themeMode} />
      </div>

      <TopBar forceRuntimeCollapsed={overlayPolicy.forceRuntimeCollapsed} />

      <div className="overlay-slot-rail pointer-events-none">
        <SideRail activePanel={activePanel} onPanelChange={setActivePanel} />
      </div>

      <ExplorerPanel
        className="overlay-slot-explorer pointer-events-auto"
        open={overlayPolicy.explorerOpen}
        view={overlayPolicy.explorerView}
        onClose={() => setActivePanel(null)}
      />

      <SettingsPanel
        open={overlayPolicy.settingsOpen}
        themeMode={themeMode}
        onThemeModeChange={setThemeMode}
        onClose={() => setActivePanel(null)}
      />

      {!overlayPolicy.suppressContextPanels && <PropertiesPanel />}
    </div>
  );
}
