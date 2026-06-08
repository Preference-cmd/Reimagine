import {
  WorkflowSchema,
  type Workflow,
  type RunId,
  type ModelInfo,
  type NodeDef,
} from "./schemas";
import {
  mockRunWorkflow,
  mockCancelRun,
  mockListModels,
  mockGetNodeDefs,
} from "./mock";

/**
 * Type-safe IPC command wrappers.
 *
 * In dev (no Rust backend yet) they call the local mock. When the Tauri
 * side is wired up (issue after the next), this file is the only place
 * that needs to swap `mockX` for `invoke("x", ...)`.
 */

const USE_MOCK = import.meta.env.DEV || import.meta.env.VITE_FORCE_MOCK === "1";

async function dispatch<TIn, TOut>(
  name: string,
  schema: { parse: (x: unknown) => TIn } | null,
  input: TIn,
  mockFn: (i: TIn) => Promise<TOut>,
): Promise<TOut> {
  // Validate input at the boundary (catches dev-time drift between
  // TS types and zod schemas).
  if (schema) schema.parse(input);

  if (USE_MOCK) {
    return mockFn(input);
  }

  // Production path — Tauri runtime. The Tauri `invoke` is dynamically
  // imported so dev builds don't pull it in.
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<TOut>(name, { input });
}

export function runWorkflow(workflow: Workflow): Promise<RunId> {
  return dispatch("run_workflow", WorkflowSchema, workflow, mockRunWorkflow);
}

export function cancelRun(runId: RunId): Promise<void> {
  return dispatch("cancel_run", null, runId, mockCancelRun);
}

export function listModels(): Promise<ModelInfo[]> {
  return dispatch("list_models", null, undefined, mockListModels);
}

export function getNodeDefs(): Promise<NodeDef[]> {
  return dispatch("get_node_defs", null, undefined, mockGetNodeDefs);
}