import React, { Suspense } from "react";
import { ErrorBoundary } from "react-error-boundary";
import { useUIStore } from "@/store/uiStore";
import { NodeCanvas } from "@/components/canvas/NodeCanvas";
import { AssetsView } from "./AssetsView";
import { SettingsView, type ThemeMode } from "./SettingsView";
import { ErrorFallback } from "./ErrorFallback";

const ModelsView = React.lazy(() =>
  import("./ModelsView").then((m) => ({ default: m.ModelsView })),
);
const RunsView = React.lazy(() => import("./RunsView").then((m) => ({ default: m.RunsView })));

function LazyFallback() {
  return (
    <div className="flex h-full w-full items-center justify-center text-muted-foreground text-sm">
      Loading...
    </div>
  );
}

/**
 * MainContent — switches between views based on active sidebar section.
 * ModelsView and RunsView are lazy-loaded to reduce the initial bundle size.
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

  switch (activeSection) {
    case "models":
      return (
        <ErrorBoundary FallbackComponent={ErrorFallback}>
          <Suspense fallback={<LazyFallback />}>
            <ModelsView />
          </Suspense>
        </ErrorBoundary>
      );
    case "runs":
      return (
        <ErrorBoundary FallbackComponent={ErrorFallback}>
          <Suspense fallback={<LazyFallback />}>
            <RunsView />
          </Suspense>
        </ErrorBoundary>
      );
    case "assets":
      return (
        <ErrorBoundary FallbackComponent={ErrorFallback}>
          <AssetsView />
        </ErrorBoundary>
      );
    case "settings":
      return (
        <ErrorBoundary FallbackComponent={ErrorFallback}>
          <SettingsView themeMode={themeMode} onThemeModeChange={onThemeModeChange} />
        </ErrorBoundary>
      );
    case "workflows":
    default:
      return (
        <ErrorBoundary FallbackComponent={ErrorFallback}>
          <NodeCanvas themeMode={themeMode} />
        </ErrorBoundary>
      );
  }
}
