import { describe, it, expect, vi, beforeEach } from "vitest";
import type { RunEvent, RunWorkflowResponse } from "../src/ipc/schemas";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

let runWorkflowDeferred = deferred<RunWorkflowResponse>();

vi.mock("@/ipc", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/ipc")>();
  return {
    ...actual,
    runWorkflow: vi.fn(() => runWorkflowDeferred.promise),
  };
});

import { useRuntimeStore } from "../src/store/runtime";
import { useArtifactStore } from "../src/store/artifacts";

function runEvent(runId: string, kind: string, nodeId: string | null = null): RunEvent {
  return {
    type: "run",
    id: `${runId}-${kind}-${nodeId ?? "run"}`,
    runId,
    workflowId: "wf-1",
    workflowVersion: 1,
    kind,
    nodeId,
    artifact: null,
    diagnostics: [],
    progress: null,
    createdAt: new Date().toISOString(),
    correlationId: null,
  };
}

function started(runId: string): RunWorkflowResponse {
  const now = new Date().toISOString();
  return {
    outcome: "started",
    runId,
    workflowId: "wf-1",
    workflowVersion: "1",
    initialSnapshot: {
      runId,
      workflowId: "wf-1",
      state: "running",
      nodeStates: {},
      diagnostics: [],
      artifacts: [],
      startedAt: now,
      updatedAt: now,
    },
    diagnostics: [],
  };
}

describe("runtime store and the event bus", () => {
  beforeEach(() => {
    useRuntimeStore.getState().reset();
    useArtifactStore.getState().reset();
    runWorkflowDeferred = deferred<RunWorkflowResponse>();
  });

  it("holds events that beat runWorkflow, then applies this run's in order", async () => {
    const run = useRuntimeStore.getState().startRun();
    const { handleEvent } = useRuntimeStore.getState();
    handleEvent(runEvent("run-1", "RunStarted"));
    handleEvent(runEvent("run-other", "NodeStarted", "node-other"));
    handleEvent(runEvent("run-1", "NodeStarted", "node-x"));
    handleEvent(runEvent("run-1", "RunCompleted"));

    expect(useRuntimeStore.getState().phase).toBe("starting");
    expect(useRuntimeStore.getState().runId).toBeNull();

    runWorkflowDeferred.resolve(started("run-1"));
    await run;

    const state = useRuntimeStore.getState();
    expect(state.runId).toBe("run-1");
    expect(state.currentNode).toBe("node-x");
    expect(state.phase).toBe("completed");
  });

  it("applies events of the active run directly and ignores other runs", async () => {
    const run = useRuntimeStore.getState().startRun();
    runWorkflowDeferred.resolve(started("run-2"));
    await run;
    expect(useRuntimeStore.getState().phase).toBe("running");

    const { handleEvent } = useRuntimeStore.getState();
    handleEvent(runEvent("run-2", "NodeStarted", "node-y"));
    handleEvent(runEvent("run-other", "RunFailed"));

    expect(useRuntimeStore.getState().currentNode).toBe("node-y");
    expect(useRuntimeStore.getState().phase).toBe("running");
  });
});
