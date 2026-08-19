import "@fontsource/geist/latin-400.css";
import "@fontsource/geist/latin-500.css";
import "@fontsource/geist/latin-600.css";
import "@fontsource/geist-mono/latin-400.css";
import "@fontsource/geist-mono/latin-500.css";

import "@fontsource/jetbrains-mono/latin-400.css";
import "@fontsource/jetbrains-mono/latin-500.css";
import "@fontsource/jetbrains-mono/latin-700.css";

import "./styles/globals.css";

import React from "react";
import ReactDOM from "react-dom/client";
import { RouterProvider, createRouter } from "@tanstack/react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { routeTree } from "./routeTree.gen";
import { ThemeProvider } from "@/lib/theme";
import { useUndoRedoShortcuts } from "@/hooks/useUndoRedo";
import { useWorkflowPersistence } from "@/hooks/useWorkflowPersistence";
import { useWorkspaceSurface } from "@/hooks/useWorkspaceSurface";
import { TooltipProvider } from "@/components/ui/tooltip";
import { setArtifactQueryClient } from "@/store/artifacts";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
});

// Wire up the artifact store to use the query cache for resolution
setArtifactQueryClient(queryClient);

const router = createRouter({
  routeTree,
  defaultPreload: "intent",
});

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

function RootEffects() {
  useUndoRedoShortcuts();
  useWorkspaceSurface();
  useWorkflowPersistence();
  return null;
}

function Root() {
  return (
    <QueryClientProvider client={queryClient}>
      <RootEffects />
      <ThemeProvider>
        <TooltipProvider delayDuration={200}>
          <RouterProvider router={router} />
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

const rootElement = document.getElementById("root")!;
ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
);
