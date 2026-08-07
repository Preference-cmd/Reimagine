import { lazy, type ComponentType } from "react";

/**
 * Route configuration — single source of truth for all views.
 *
 * Each route defines:
 * - `id`: Unique identifier matching SidebarSection
 * - `component`: Lazy-loaded React component
 * - `showRunFab`: Whether to show the floating action button
 * - `showPropertiesDrawer`: Whether the properties drawer is available
 */

export type RouteId = "workflows" | "models" | "runs" | "assets" | "settings";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type AnyComponent = ComponentType<any>;

export interface RouteConfig {
  id: RouteId;
  component: React.LazyExoticComponent<AnyComponent>;
  showRunFab?: boolean;
  showPropertiesDrawer?: boolean;
}

// Lazy-loaded view components
const NodeCanvas = lazy(() =>
  import("@/components/canvas/NodeCanvas").then((m) => ({ default: m.NodeCanvas })),
);

const ModelsView = lazy(() =>
  import("@/components/layout/ModelsView").then((m) => ({ default: m.ModelsView })),
);

const RunsView = lazy(() =>
  import("@/components/layout/RunsView").then((m) => ({ default: m.RunsView })),
);

const AssetsView = lazy(() =>
  import("@/components/layout/AssetsView").then((m) => ({ default: m.AssetsView })),
);

const SettingsView = lazy(() =>
  import("@/components/layout/SettingsView").then((m) => ({ default: m.SettingsView })),
);

/**
 * Route registry — ordered by navigation appearance.
 * Add new views here; MainContent renders them automatically.
 */
export const ROUTES: RouteConfig[] = [
  {
    id: "workflows",
    component: NodeCanvas,
    showRunFab: true,
    showPropertiesDrawer: true,
  },
  {
    id: "models",
    component: ModelsView,
  },
  {
    id: "runs",
    component: RunsView,
  },
  {
    id: "assets",
    component: AssetsView,
  },
  {
    id: "settings",
    component: SettingsView,
  },
];

/**
 * Get route config by ID.
 * Returns the workflows route as fallback for unknown IDs.
 */
export function getRoute(id: string | null): RouteConfig {
  return ROUTES.find((r) => r.id === id) ?? ROUTES[0];
}

/**
 * Check if a route ID is valid.
 */
export function isValidRoute(id: string | null): id is RouteId {
  return ROUTES.some((r) => r.id === id);
}
