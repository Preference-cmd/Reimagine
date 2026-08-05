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
import App from "./App";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useUndoRedoShortcuts } from "@/hooks/useUndoRedo";
import { useWorkflowPersistence } from "@/hooks/useWorkflowPersistence";

function Root() {
  useUndoRedoShortcuts();
  useWorkflowPersistence();
  return <App />;
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <TooltipProvider delayDuration={200}>
      <Root />
    </TooltipProvider>
  </React.StrictMode>,
);
