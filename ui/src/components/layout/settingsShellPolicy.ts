import {
  isSettingsPanel,
  getOverlayPolicy,
  type OverlayPanel,
} from "./overlayLayout";

export type ShellPanel = OverlayPanel;

export { isSettingsPanel };

export function shouldSuppressContextPanels(panel: ShellPanel) {
  return getOverlayPolicy(panel).suppressContextPanels;
}

export function shouldForceRuntimeCollapsed(panel: ShellPanel) {
  return getOverlayPolicy(panel).forceRuntimeCollapsed;
}
