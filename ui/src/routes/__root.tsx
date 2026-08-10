import { createRootRoute, Outlet } from "@tanstack/react-router";
import { ErrorBoundary } from "react-error-boundary";
import { Toaster } from "sonner";
import { Sidebar } from "@/components/layout/Sidebar";
import { WindowControls } from "@/components/layout/WindowControls";
import { PropertiesDrawer } from "@/components/layout/PropertiesDrawer";
import { RunFab } from "@/components/layout/RunFab";
import { ErrorFallback } from "@/components/layout/ErrorFallback";
import { ContextMenuPanel } from "@/components/canvas/ContextMenuPanel";
import { RenameNodeDialog } from "@/components/canvas/RenameNodeDialog";
import { CommandPalette } from "@/components/palette/CommandPalette";
import { useNodeRegistryStore } from "@/store/nodeRegistry";
import { useTheme } from "@/lib/theme";
import { useEffect } from "react";
import { useNodeDefs } from "@/hooks/queries";

function RootLayout() {
  const { themeMode } = useTheme();
  const isWorkflows = false; // /new is the welcome screen, not the workflow editor

  // Node catalog: fetch once via TanStack Query, hydrate the Zustand store
  const nodeDefsQuery = useNodeDefs();
  useEffect(() => {
    if (nodeDefsQuery.data) {
      useNodeRegistryStore.getState().hydrate(nodeDefsQuery.data);
    } else if (nodeDefsQuery.error) {
      useNodeRegistryStore.getState().hydrate([], nodeDefsQuery.error.message);
    }
  }, [nodeDefsQuery.data, nodeDefsQuery.error]);

  return (
    <div className="app-shell flex h-full min-w-screen w-full bg-background text-foreground">
      <Sidebar />

      <main className="main-content relative flex min-w-0 flex-1 flex-col overflow-clip">
        {isWorkflows && <WindowControls />}
        <div className="relative min-h-0 flex-1">
          <ErrorBoundary FallbackComponent={ErrorFallback}>
            <Outlet />
          </ErrorBoundary>
          {isWorkflows && <RunFab />}
        </div>
      </main>

      {isWorkflows && <PropertiesDrawer />}

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

export const Route = createRootRoute({
  component: RootLayout,
});
