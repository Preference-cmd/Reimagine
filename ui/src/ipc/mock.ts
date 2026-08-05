import {
  type ArtifactMetadata,
  type DownloadEventPayload,
  type DownloadHuggingfaceModelArgs,
  type ModelCard,
  type ModelCatalogEntry,
  type ModelDownloadOutput,
  type ModelInfo,
  type NodeDef,
  type RunWorkflowResponse,
  type Workflow,
  type WorkflowFileSummary,
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

/* ───── Workflow persistence mocks (localStorage-backed) ─────
   Kept in localStorage so dev sessions survive reloads, mirroring the
   on-disk `workflows/` behavior of the real backend. */

const PERSISTENCE_KEY = "reimagine.mock.workflows";

function mockReadAll(): Record<string, string> {
  try {
    return JSON.parse(localStorage.getItem(PERSISTENCE_KEY) ?? "{}");
  } catch {
    return {};
  }
}

function mockWriteAll(all: Record<string, string>): void {
  localStorage.setItem(PERSISTENCE_KEY, JSON.stringify(all));
}

export async function mockSaveWorkflow(input: {
  workflowId: string;
  workflowJson: unknown;
}): Promise<string> {
  await delay(50);
  const all = mockReadAll();
  all[input.workflowId] = JSON.stringify(input.workflowJson);
  mockWriteAll(all);
  return `mock:/workflows/${input.workflowId}.json`;
}

export async function mockLoadWorkflow(input: {
  workflowId: string;
}): Promise<unknown> {
  await delay(50);
  const raw = mockReadAll()[input.workflowId];
  if (!raw) {
    throw new Error(`mock workflow not found: ${input.workflowId}`);
  }
  return JSON.parse(raw);
}

export async function mockListWorkflows(): Promise<WorkflowFileSummary[]> {
  await delay(30);
  const all = mockReadAll();
  return Object.entries(all).map(([id], index, entries) => ({
    id,
    // Entries are in insertion order, so the last-saved workflow gets the
    // largest timestamp and sorts first (newest-first), like the backend.
    modified_millis: entries.length - index,
  }));
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

export async function mockResolveArtifact(
  artifactId: string,
): Promise<ArtifactMetadata> {
  await delay(100);
  return {
    id: artifactId,
    nodeId: "node-save-image",
    mediaType: "image/png",
    filename: `${artifactId}.png`,
    path: `/workspace/output/${artifactId}.png`,
  };
}

export async function mockOpenArtifact(_artifactId: string): Promise<void> {
  await delay(100);
}

const MOCK_CATALOG: ModelCatalogEntry[] = [
  {
    id: "stabilityai/stable-diffusion-xl-base-1.0",
    author: "stabilityai",
    pipelineTag: "text-to-image",
    tags: ["diffusers", "safetensors"],
    downloads: 12_400_000,
    likes: 10_200,
    lastModified: "2024-01-15T00:00:00Z",
    private: false,
  },
  {
    id: "runwayml/stable-diffusion-v1-5",
    author: "runwayml",
    pipelineTag: "text-to-image",
    tags: ["diffusers", "safetensors"],
    downloads: 25_100_000,
    likes: 21_300,
    lastModified: "2023-10-01T00:00:00Z",
    private: false,
  },
];

const MOCK_CARD: ModelCard = {
  entry: MOCK_CATALOG[0],
  detectedFormat: "Diffusers",
  estimatedDownloadSize: 6_940_000_000,
  modelSummary:
    "Stable Diffusion XL base is a latent diffusion model for text-to-image generation.",
  fileCount: 12,
  components: ["unet", "text_encoder", "text_encoder_2", "vae"],
};

export async function mockSearchModels(
  input: { query: string; filters?: unknown },
): Promise<ModelCatalogEntry[]> {
  await delay(200);
  const needle = input.query.trim().toLowerCase();
  if (!needle) return [];
  return MOCK_CATALOG.filter((entry) => entry.id.toLowerCase().includes(needle));
}

export async function mockGetModelCard(_input: {
  repoId: string;
}): Promise<ModelCard> {
  await delay(150);
  return MOCK_CARD;
}

export async function mockDownloadHuggingfaceModel(
  args: DownloadHuggingfaceModelArgs,
  onEvent?: (event: DownloadEventPayload) => void,
): Promise<ModelDownloadOutput> {
  const total = 6_940_000_000;
  const chunk = 500_000_000;
  let bytes = 0;
  const id = rand("dl");
  const repoId = args.repoId;

  const emit = (status: string, extra: Partial<DownloadEventPayload> = {}) => {
    onEvent?.({
      id,
      status,
      repoId,
      revision: args.revision ?? "main",
      bytesDownloaded: 0,
      ...extra,
    });
  };

  emit("started", { totalBytes: total, message: "Fetching metadata" });
  while (bytes < total) {
    await delay(80);
    bytes += chunk;
    emit("in_progress", {
      bytesDownloaded: Math.min(chunk, total - bytes + chunk),
      totalBytes: total,
      message: "Downloading model files",
    });
  }
  emit("completed", { bytesDownloaded: total, totalBytes: total });

  return {
    effective: true,
    provider: "hf",
    repoId,
    revision: args.revision ?? "main",
    targetDir: args.targetRelativeDir,
    files: [{ relativePath: "model_index.json", bytes: total, outcome: "downloaded" }],
    totalBytes: total,
    finishedAt: new Date().toISOString(),
    detectedFormat: "Diffusers",
  };
}
