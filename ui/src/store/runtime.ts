import { create } from "zustand";
import type { RunEvent } from "@/ipc/schemas";
import { runWorkflow as ipcRunWorkflow, cancelRun as ipcCancelRun, subscribeEvents } from "@/ipc";
import { useArtifactStore } from "./artifacts";
import { useProjectStore } from "./project";

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
  /** Feed a run event from the event bus. */
  handleEvent: (event: RunEvent) => void;
};

/** Run events that arrived while `runWorkflow` had not yet returned the run
    id; `null` when no start is in flight. The run's first events can beat the
    response, so they wait here. This is separate from `phase`, which run
    events themselves change (`RunQueued` maps to "starting"). */
let pendingEvents: RunEvent[] | null = null;

export const useRuntimeStore = create<RuntimeState>()((set, get) => ({
  phase: "idle",
  runId: null,
  workflowName: "",
  backend: "Burn",
  device: "CPU",
  currentNode: null,
  progress: 0,
  elapsedMs: 0,
  diagnostics: [],

  handleEvent: (event: RunEvent) => {
    if (pendingEvents) {
      pendingEvents.push(event);
      return;
    }
    if (event.runId !== get().runId) return;

    // F5-4: artifact/run lifecycle events drive the inline preview cache.
    useArtifactStore.getState().handleEvent(event);

    const next = eventKindToPhase(event.kind);
    set((s) => ({ currentNode: event.nodeId ?? s.currentNode, phase: next }));
    // Terminal states stop the clock — no more elapsedMs/progress ticks
    // after the run finishes.
    if (isTerminalPhase(next)) {
      stopProgressTimer();
    }
  },

  startRun: async (workflowJson?: unknown) => {
    // Defensive: a previous run's timer must never outlive a new run.
    stopProgressTimer();
    pendingEvents = [];
    set({
      phase: "starting",
      runId: null,
      currentNode: null,
      diagnostics: [],
      progress: 0,
      elapsedMs: 0,
    });

    const startedAt = Date.now();
    activeProgressTimer = setInterval(() => {
      set((s) => ({
        elapsedMs: Date.now() - startedAt,
        ...(s.phase === "running" ? { progress: Math.min(1, s.progress + 0.02) } : {}),
      }));
    }, 500);

    try {
      const response = await ipcRunWorkflow(
        useProjectStore.getState().activeProjectId,
        workflowJson as any,
      );
      const early = pendingEvents ?? [];
      pendingEvents = null;

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
        const { handleEvent } = get();
        for (const event of early) {
          if (event.runId === response.runId) handleEvent(event);
        }
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
      pendingEvents = null;
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

subscribeEvents((event) => {
  if (event.type === "run") useRuntimeStore.getState().handleEvent(event);
});
