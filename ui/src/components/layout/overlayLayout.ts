export type OverlayPanel = string | null;

export const EXPLORER_PANELS = ["Graph", "Models", "Runs", "Assets"] as const;

export type ExplorerPanelId = (typeof EXPLORER_PANELS)[number];

export type OverlayPolicy = {
  explorerOpen: boolean;
  explorerView: ExplorerPanelId | null;
  forceRuntimeCollapsed: boolean;
  settingsOpen: boolean;
  suppressContextPanels: boolean;
};

export function isExplorerPanel(panel: OverlayPanel): panel is ExplorerPanelId {
  return EXPLORER_PANELS.includes(panel as ExplorerPanelId);
}

export function isSettingsPanel(panel: OverlayPanel): panel is "Settings" {
  return panel === "Settings";
}

export function getOverlayPolicy(panel: OverlayPanel): OverlayPolicy {
  const settingsOpen = isSettingsPanel(panel);
  const explorerOpen = isExplorerPanel(panel);

  return {
    explorerOpen,
    explorerView: explorerOpen ? panel : null,
    forceRuntimeCollapsed: settingsOpen,
    settingsOpen,
    suppressContextPanels: settingsOpen,
  };
}
