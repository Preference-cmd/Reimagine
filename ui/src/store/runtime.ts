import { create } from "zustand";
import type { RunEventPayload } from "@/ipc/schemas";
import { runWorkflow as ipcRunWorkflow, cancelRun as ipcCancelRun } from "@/ipc";
import { useArtifactStore } from "./artifacts";

export type RuntimePhase = "idle" | "starting" | "running" | "completed" | "failed" | "cancelled";

export type RuntimeDiagnostic = {
  id: string;
  severity: string;
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
  startRun: (workflowJson?: unknown) => Promise<void>;
  cancelRun: () => Promise<void>;
  reset: () => void;
};

export const useRuntimeStore = create<RuntimeState>()((set, get) => ({
  phase: "idle",
  runId: null,
  workflowName: "",
  backend: "Candle",
  device: "CPU",
  currentNode: null,
  progress: 0,
  elapsedMs: 0,
  diagnostics: [],

  startRun: async (workflowJson?: unknown) => {
    // Defensive: a previous run's timer must never outlive a new run.
    stopProgressTimer();
    set({ phase: "starting", diagnostics: [], progress: 0, elapsedMs: 0 });

    const startedAt = Date.now();
    activeProgressTimer = setInterval(() => {
      set((s) => ({
        elapsedMs: Date.now() - startedAt,
        ...(s.phase === "running" ? { progress: Math.min(1, s.progress + 0.02) } : {}),
      }));
    }, 500);

    const handleEvent = (event: RunEventPayload) => {
      // F5-4: artifact/run lifecycle events drive the inline preview cache.
      useArtifactStore.getState().handleEvent(event);

      const phase = eventKindToPhase(event.kind);
      set((s) => ({
        runId: event.runId,
        currentNode: event.nodeId ?? s.currentNode,
        phase,
      }));
      // Terminal states stop the clock — no more elapsedMs/progress ticks
      // after the run finishes.
      if (isTerminalPhase(phase)) {
        stopProgressTimer();
      }
    };

    try {
      const response = await ipcRunWorkflow(workflowJson as any, handleEvent);

      if (response.outcome === "started") {
        set({
          runId: response.runId,
          phase: "running",
          diagnostics: response.diagnostics.map((d) => ({
            id: d.id,
            severity: d.severity,
            message: d.message,
            source: d.source,
          })),
        });
      } else {
        set({
          phase: "failed",
          diagnostics: response.diagnostics.map((d) => ({
            id: d.id,
            severity: d.severity,
            message: d.message,
            source: d.source,
          })),
        });
        stopProgressTimer();
      }
    } catch (err) {
      set({
        phase: "failed",
        diagnostics: [
          {
            id: "run-error",
            severity: "error",
            message: String(err),
            source: "Runtime",
          },
        ],
      });
      stopProgressTimer();
    }
  },

  cancelRun: async () => {
    const { runId } = get();
    if (!runId) return;
    try {
      await ipcCancelRun(runId);
      stopProgressTimer();
      set({ phase: "cancelled" });
    } catch {
      // If cancel fails, just reset
      set({
        phase: "idle",
        runId: null,
        currentNode: null,
        progress: 0,
        elapsedMs: 0,
        diagnostics: [],
      });
    }
  },

  reset: () => {
    stopProgressTimer();
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

function eventKindToPhase(kind: string): RuntimePhase {
  switch (kind) {
    case "RunQueued":
      return "starting";
    case "RunStarted":
      return "running";
    case "RunCompleted":
      return "completed";
    case "RunFailed":
      return "failed";
    case "RunCancelled":
      return "cancelled";
    default:
      return "running";
  }
}

function isTerminalPhase(phase: RuntimePhase): boolean {
  return phase === "completed" || phase === "failed" || phase === "cancelled";
}

/** Module-level timer guard so every terminal path (events, cancel, reset,
    errors) can stop the interval, not just the callers that own the handle. */
let activeProgressTimer: ReturnType<typeof setInterval> | null = null;

function stopProgressTimer() {
  if (activeProgressTimer !== null) {
    clearInterval(activeProgressTimer);
    activeProgressTimer = null;
  }
}
