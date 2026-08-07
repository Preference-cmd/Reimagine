import { useCallback, useMemo } from "react";
import { useUIStore } from "@/store/uiStore";
import { getRoute, isValidRoute, type RouteId } from "@/lib/routes";

// Re-export RouteId for consumers
export type { RouteId } from "@/lib/routes";

/**
 * Navigation hook — encapsulates all routing logic.
 *
 * Provides:
 * - Current route ID and config
 * - Navigation actions
 * - Route validation
 */
export function useNavigation() {
  const activeSection = useUIStore((s) => s.activeSidebarSection);
  const setActiveSection = useUIStore((s) => s.setActiveSidebarSection);
  const settingsNavId = useUIStore((s) => s.settingsNavId);

  // Current route config
  const currentRoute = useMemo(() => getRoute(activeSection), [activeSection]);

  // Navigation actions
  const navigate = useCallback(
    (routeId: RouteId) => {
      if (isValidRoute(routeId)) {
        setActiveSection(routeId);
      }
    },
    [setActiveSection],
  );

  const navigateToWorkflows = useCallback(() => navigate("workflows"), [navigate]);
  const navigateToModels = useCallback(() => navigate("models"), [navigate]);
  const navigateToRuns = useCallback(() => navigate("runs"), [navigate]);
  const navigateToAssets = useCallback(() => navigate("assets"), [navigate]);
  const navigateToSettings = useCallback(() => navigate("settings"), [navigate]);

  // Check if a specific route is active
  const isRouteActive = useCallback(
    (routeId: RouteId) => activeSection === routeId,
    [activeSection],
  );

  // Check if we're in settings (for sidebar mode switching)
  const isSettings = activeSection === "settings";

  return {
    // Current state
    activeSection,
    currentRoute,
    isSettings,
    settingsNavId,

    // Navigation
    navigate,
    navigateToWorkflows,
    navigateToModels,
    navigateToRuns,
    navigateToAssets,
    navigateToSettings,

    // Helpers
    isRouteActive,
    isValidRoute,
  };
}
