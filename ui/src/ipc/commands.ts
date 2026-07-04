import {
  type ModelInfo,
  type NodeDef,
  type RunEventPayload,
  type RunWorkflowResponse,
  type Workflow,
} from "./schemas";
import {
  mockCancelRun,
  mockGetNodeDefs,
  mockListModels,
  mockRunWorkflow,
} from "./mock";

const USE_MOCK = import.meta.env.DEV || import.meta.env.VITE_FORCE_MOCK === "1";

async function dispatch<TIn, TOut>(
  name: string,
  schema: { parse: (x: unknown) => TIn } | null,
  input: TIn,
  mockFn: (i: TIn) => Promise<TOut>,
): Promise<TOut> {
  if (schema) schema.parse(input);
  if (USE_MOCK) return mockFn(input);
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<TOut>(name, { input });
}

export async function runWorkflow(
  workflow: Workflow,
  onEvent?: (event: RunEventPayload) => void,
): Promise<RunWorkflowResponse> {
  if (USE_MOCK) {
    return mockRunWorkflow(workflow);
  }

  const { Channel, invoke } = await import("@tauri-apps/api/core");
  const channel = new Channel<RunEventPayload>();
  if (onEvent) {
    channel.onmessage = onEvent;
  }
  return invoke<RunWorkflowResponse>("run_workflow", { workflow, channel });
}

export async function cancelRun(runId: string): Promise<void> {
  if (USE_MOCK) {
    return mockCancelRun(runId);
  }
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<void>("cancel_run", { runId });
}

export function listModels(): Promise<ModelInfo[]> {
  return dispatch("list_models", null, undefined, mockListModels);
}

export function getNodeDefs(): Promise<NodeDef[]> {
  return dispatch("get_node_defs", null, undefined, mockGetNodeDefs);
}
