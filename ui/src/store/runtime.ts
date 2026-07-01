import { create } from "zustand";

export type RuntimePhase =
  | "idle"
  | "starting"
  | "running"
  | "completed"
  | "failed";

export type RuntimeDiagnosticSeverity = "info" | "warning" | "error";

export type RuntimeDiagnostic = {
  id: string;
  severity: RuntimeDiagnosticSeverity;
  message: string;
  source: string;
};

type RuntimeState = {
  phase: RuntimePhase;
  runId: string | null;
  workflowName: string;
  backend: string;
  device: string;
  currentNode: string | null;
  progress: number;
  elapsedMs: number;
  diagnostics: RuntimeDiagnostic[];
  startMockRun: () => void;
  cancelRun: () => void;
};

let phaseTimer: ReturnType<typeof setTimeout> | null = null;
let progressTimer: ReturnType<typeof setInterval> | null = null;

function clearMockTimers() {
  if (phaseTimer) {
    clearTimeout(phaseTimer);
    phaseTimer = null;
  }
  if (progressTimer) {
    clearInterval(progressTimer);
    progressTimer = null;
  }
}

function nodeForProgress(progress: number) {
  if (progress < 0.22) return "Model";
  if (progress < 0.48) return "Positive prompt";
  if (progress < 0.78) return "Sampler";
  return "Image output";
}

function runId() {
  return `run_${Math.random().toString(36).slice(2, 10)}`;
}

export const useRuntimeStore = create<RuntimeState>()((set, get) => ({
  phase: "idle",
  runId: null,
  workflowName: "Black bear",
  backend: "Candle",
  device: "Metal",
  currentNode: null,
  progress: 0,
  elapsedMs: 0,
  diagnostics: [],

  startMockRun: () => {
    clearMockTimers();

    const startedAt = Date.now();
    const id = runId();

    set({
      phase: "starting",
      runId: id,
      currentNode: "Readiness",
      progress: 0,
      elapsedMs: 0,
      diagnostics: [
        {
          id: "runtime-ready",
          severity: "info",
          source: "Runtime",
          message: "Graph accepted. Preparing execution plan.",
        },
      ],
    });

    phaseTimer = setTimeout(() => {
      set({
        phase: "running",
        currentNode: "Model",
        diagnostics: [],
      });

      progressTimer = setInterval(() => {
        const state = get();
        if (state.phase !== "running") return;

        const nextProgress = Math.min(1, state.progress + 0.055);
        const nextElapsed = Date.now() - startedAt;

        if (nextProgress >= 1) {
          clearMockTimers();
          set({
            phase: "completed",
            currentNode: "Image output",
            progress: 1,
            elapsedMs: nextElapsed,
            diagnostics: [
              {
                id: "run-complete",
                severity: "info",
                source: "Runtime",
                message: "Workflow finished without blocking diagnostics.",
              },
            ],
          });
          return;
        }

        set({
          progress: nextProgress,
          elapsedMs: nextElapsed,
          currentNode: nodeForProgress(nextProgress),
        });
      }, 420);
    }, 650);
  },

  cancelRun: () => {
    clearMockTimers();
    set({
      phase: "idle",
      runId: null,
      currentNode: null,
      progress: 0,
      elapsedMs: 0,
      diagnostics: [],
    });
  },
}));
