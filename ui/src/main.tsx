import "@fontsource/geist/latin-400.css";
import "@fontsource/geist/latin-500.css";
import "@fontsource/geist/latin-600.css";
import "@fontsource/geist-mono/latin-400.css";
import "@fontsource/geist-mono/latin-500.css";

import "@fontsource/inter/latin-400.css";
import "@fontsource/inter/latin-500.css";
import "@fontsource/inter/latin-600.css";
import "@fontsource/inter/latin-700.css";
import "@fontsource/jetbrains-mono/latin-400.css";
import "@fontsource/jetbrains-mono/latin-500.css";
import "@fontsource/jetbrains-mono/latin-700.css";

import "./styles/globals.css";

import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useUndoRedoShortcuts } from "@/hooks/useUndoRedo";
import { runWorkflow, type Workflow } from "@/ipc";

function Root() {
  useUndoRedoShortcuts();

  // Smoke test the IPC wrapper: just call it once on mount to ensure
  // the mock + zod roundtrip works. The result is logged for the dev.
  React.useEffect(() => {
    const wf: Workflow = { nodes: [], edges: [] };
    runWorkflow(wf)
      .then((runId) => console.info("[ipc] mock runWorkflow ->", runId))
      .catch((err) => console.error("[ipc] mock runWorkflow failed:", err));
  }, []);

  return <App />;
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <TooltipProvider delayDuration={200}>
      <Root />
    </TooltipProvider>
  </React.StrictMode>,
);
