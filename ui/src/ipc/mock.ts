import {
  type ModelInfo,
  type NodeDef,
  type RunWorkflowResponse,
  type Workflow,
  NodeDefSchema,
} from "./schemas";

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

function rand(prefix: string) {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

const MOCK_MODELS: ModelInfo[] = [
  {
    id: "sd_xl_base_1_0",
    displayName: "Stable Diffusion Xl Base",
    modelSeries: "stable-diffusion-xl",
    variant: "base",
    roles: ["checkpoint-bundle", "diffusion-model"],
    format: "safetensors",
    sourceStatus: "available",
    sizeBytes: 6_940_000_000,
  },
  {
    id: "dreamshaper_8",
    displayName: "Stable Diffusion 1.5 Dreamshaper",
    modelSeries: "stable-diffusion-1.5",
    variant: "dreamshaper",
    roles: ["checkpoint-bundle"],
    format: "safetensors",
    sourceStatus: "available",
    sizeBytes: 2_070_000_000,
  },
];

export async function mockRunWorkflow(_workflow: Workflow): Promise<RunWorkflowResponse> {
  await delay(200);
  const runId = rand("run");
  return {
    outcome: "started",
    runId,
    workflowId: rand("wf"),
    workflowVersion: "1",
    initialSnapshot: {
      runId,
      workflowId: "mock-wf",
      state: "running",
      nodeStates: {},
      diagnostics: [],
      artifacts: [],
      startedAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    },
    diagnostics: [],
  };
}

export async function mockCancelRun(_runId: string): Promise<void> {
  await delay(100);
}

export async function mockListModels(): Promise<ModelInfo[]> {
  await delay(150);
  return [...MOCK_MODELS];
}

export async function mockGetNodeDefs(): Promise<NodeDef[]> {
  const response = await fetch("/nodes").catch(() => null);
  if (!response?.ok) return [];
  const payload = await response.json();
  const nodes: unknown[] = Array.isArray(payload?.nodes) ? payload.nodes : [];
  return nodes.map((node) => NodeDefSchema.parse(node));
}
