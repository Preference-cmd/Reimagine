import { Suspense } from "react";
import { ErrorBoundary } from "react-error-boundary";
import { getRoute } from "@/lib/routes";
import { useUIStore } from "@/store/uiStore";
import { ErrorFallback } from "./ErrorFallback";
import type { ThemeMode } from "./SettingsView";

function LazyFallback() {
  return (
    <div className="flex h-full w-full items-center justify-center text-muted-foreground text-sm">
      Loading...
    </div>
  );
}

/**
 * MainContent — switches between views based on active sidebar section.
 * Uses route configuration for automatic view rendering and lazy loading.
 * Each view is wrapped in its own ErrorBoundary so a crash in one view
 * does not bring down the rest of the application.
 */
export function MainContent({
  themeMode,
  onThemeModeChange,
}: {
  themeMode: ThemeMode;
  onThemeModeChange: (mode: ThemeMode) => void;
}) {
  const activeSection = useUIStore((s) => s.activeSidebarSection);
  const route = getRoute(activeSection);
  const ViewComponent = route.component;

  // Build props based on route ID — avoids type gymnastics
  const props =
    route.id === "workflows"
      ? { themeMode }
      : route.id === "settings"
        ? { themeMode, onThemeModeChange }
        : {};

  return (
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      <Suspense fallback={<LazyFallback />}>
        <ViewComponent {...props} />
      </Suspense>
    </ErrorBoundary>
  );
}
