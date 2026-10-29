import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { useCollectionRun } from "../src/hooks/useCollectionRun";
import * as ipc from "../src/ipc";
import type { DynamicWorkflowRun, InvokeResult } from "../src/ipc/schemas";

type Listener = (event: unknown) => void;

function runDocument(
  id: string,
  state: "running" | "completed",
  version: number,
): DynamicWorkflowRun {
  return {
    id,
    project_id: "default",
    dynamic_workflow_id: "dwf-1",
    target: { collection_id: "col-1" },
    state,
    version,
    items: [
      {
        item_id: "s1",
        slot_id: "s1",
        state: state === "completed" ? "succeeded" : "running",
      },
    ],
    created_at: "2026-01-01T00:00:00Z",
  };
}

function listed(...runs: DynamicWorkflowRun[]): InvokeResult {
  return {
    step: {} as never,
    output: { runs },
  };
}

function sleep(ms: number) {
  return new Promise<void>((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

describe("useCollectionRun", () => {
  let listeners: Listener[];

  beforeEach(() => {
    listeners = [];
    vi.spyOn(ipc, "subscribeEvents").mockImplementation((listener) => {
      const captured = listener as unknown as Listener;
      listeners.push(captured);
      return () => {
        const index = listeners.indexOf(captured);
        if (index >= 0) listeners.splice(index, 1);
      };
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  function emitDocument(event: Record<string, unknown>) {
    act(() => {
      for (const listener of [...listeners]) {
        listener({
          type: "document",
          projectId: "default",
          ...event,
        });
      }
    });
  }

  test("discovers a new run announced by an event", async () => {
    let listCalls = 0;
    vi.spyOn(ipc, "opInvoke").mockImplementation(async (_project, name, input) => {
      expect(name).toBe("dynamic_workflow.list_runs");
      // Only the newest run is shown, so only it is read.
      expect(input).toEqual({ collection_id: "col-1", limit: 1 });
      listCalls += 1;
      return listCalls === 1
        ? listed(runDocument("run-1", "running", 1))
        : listed(runDocument("run-2", "running", 1));
    });

    const { result } = renderHook(() => useCollectionRun("default", "col-1"));
    await waitFor(() => expect(result.current.run?.id).toBe("run-1"));
    expect(listCalls).toBe(1);

    // The event names a run the hook has never seen: it may be this
    // collection's newest run, so it must trigger a read.
    emitDocument({
      kind: "dynamic_workflow.run_changed",
      documentId: "run-2",
      version: 1,
    });
    await waitFor(() => expect(result.current.run?.id).toBe("run-2"));
    expect(listCalls).toBe(2);
  });

  test("discovers the first run when the initial read found none", async () => {
    let listCalls = 0;
    vi.spyOn(ipc, "opInvoke").mockImplementation(async () => {
      listCalls += 1;
      return listCalls === 1 ? listed() : listed(runDocument("run-1", "running", 1));
    });

    const { result } = renderHook(() => useCollectionRun("default", "col-1"));
    await waitFor(() => expect(listCalls).toBe(1));
    expect(result.current.run).toBeNull();

    emitDocument({
      kind: "dynamic_workflow.slot_changed",
      documentId: "run-1",
      version: 1,
    });
    await waitFor(() => expect(result.current.run?.id).toBe("run-1"));
    expect(listCalls).toBe(2);
  });

  test("ignores events from another project or of an unrelated kind", async () => {
    let listCalls = 0;
    vi.spyOn(ipc, "opInvoke").mockImplementation(async () => {
      listCalls += 1;
      return listed(runDocument("run-1", "running", 1));
    });

    const { result } = renderHook(() => useCollectionRun("default", "col-1"));
    await waitFor(() => expect(result.current.run?.id).toBe("run-1"));
    expect(listCalls).toBe(1);

    emitDocument({
      projectId: "other",
      kind: "dynamic_workflow.slot_changed",
      documentId: "run-2",
      version: 1,
    });
    emitDocument({
      kind: "collection.changed",
      documentId: "run-2",
      version: 1,
    });
    await sleep(80);

    expect(listCalls, "unrelated events must not trigger a read").toBe(1);
    expect(result.current.run?.id).toBe("run-1");
  });

  test("coalesces a burst of run events into one read", async () => {
    let listCalls = 0;
    vi.spyOn(ipc, "opInvoke").mockImplementation(async () => {
      listCalls += 1;
      return listed(runDocument("run-1", "running", listCalls));
    });

    const { result } = renderHook(() => useCollectionRun("default", "col-1"));
    await waitFor(() => expect(result.current.run?.id).toBe("run-1"));
    expect(listCalls).toBe(1);

    emitDocument({
      kind: "dynamic_workflow.slot_changed",
      documentId: "run-1",
      version: 2,
    });
    emitDocument({
      kind: "dynamic_workflow.run_changed",
      documentId: "run-1",
      version: 3,
    });
    expect(listCalls, "reads are deferred by the debounce").toBe(1);

    await act(async () => {
      await sleep(80);
    });
    expect(listCalls).toBe(2);
  });

  test("a stale response cannot roll the run state back", async () => {
    const pending: Array<(value: InvokeResult) => void> = [];
    let listCalls = 0;
    vi.spyOn(ipc, "opInvoke").mockImplementation(() => {
      listCalls += 1;
      if (listCalls === 1) {
        return Promise.resolve(listed(runDocument("run-1", "running", 1)));
      }
      return new Promise((resolve) => {
        pending.push(resolve);
      });
    });

    const { result } = renderHook(() => useCollectionRun("default", "col-1"));
    await waitFor(() => expect(result.current.run?.id).toBe("run-1"));

    emitDocument({
      kind: "dynamic_workflow.slot_changed",
      documentId: "run-1",
      version: 2,
    });
    await sleep(70);
    expect(listCalls).toBe(2);

    emitDocument({
      kind: "dynamic_workflow.run_changed",
      documentId: "run-1",
      version: 3,
    });
    await sleep(70);
    expect(listCalls).toBe(3);

    // The newer read settles first, then the older one resolves late.
    await act(async () => {
      pending[1](listed(runDocument("run-1", "completed", 3)));
    });
    await waitFor(() => expect(result.current.run?.state).toBe("completed"));

    await act(async () => {
      pending[0](listed(runDocument("run-1", "running", 2)));
    });
    await sleep(10);

    expect(result.current.run?.state).toBe("completed");
    expect(result.current.slotStates.s1).toBe("succeeded");
  });
});
