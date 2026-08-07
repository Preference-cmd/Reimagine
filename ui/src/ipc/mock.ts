import {
  type ArtifactMetadata,
  type ComputeProfile,
  type DownloadEventPayload,
  type DownloadHuggingfaceModelArgs,
  type ModelCard,
  type ModelCatalogEntry,
  type ModelDownloadOutput,
  type ModelInfo,
  type NodeDef,
  type RebootBackendArgs,
  type RunEventPayload,
  type RunWorkflowResponse,
  type Workflow,
  type WorkflowFileSummary,
  type WorkerSwitchArgs,
  type WorkerSwitchResult,
  type WorkerSwitchTarget,
  NodeDefSchema,
} from "./schemas";
import { useWorkflowStore } from "@/store/workflow";

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

export async function mockRunWorkflow(
  workflow: Workflow | undefined,
  onEvent?: (event: RunEventPayload) => void,
): Promise<RunWorkflowResponse> {
  await delay(200);
  const runId = rand("run");

  const emit = (kind: string, nodeId?: string | null, artifactId?: string | null) =>
    onEvent?.({
      id: rand("evt"),
      runId,
      kind,
      nodeId: nodeId ?? null,
      artifactId: artifactId ?? null,
      createdAt: new Date().toISOString(),
    });

  // Demo event stream so inline previews (F5-4) are visible in dev even
  // when the caller passed no workflow payload (TopBar runs without args).
  const nodes = workflow?.nodes?.length ? workflow.nodes : useWorkflowStore.getState().nodes;

  emit("RunStarted");
  for (const node of nodes) {
    const type = node.type ?? (node as { type_id?: string }).type_id ?? "";
    emit("NodeStarted", node.id);
    await delay(350);
    if (PRODUCES_PREVIEW.has(type)) {
      const artifactId = `${runId}-${node.id}-0`;
      emit("ArtifactCreated", node.id, artifactId);
      await delay(250);
      emit("PreviewUpdated", node.id, artifactId);
    }
    await delay(250);
    emit("NodeCompleted", node.id);
  }
  emit("RunCompleted");

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

/** Node types that emit image artifacts in the mock run. */
const PRODUCES_PREVIEW = new Set([
  "builtin.preview_image",
  "builtin.save_image",
  "builtin.vae_decode",
  "imageGenerator",
  "imageOutput",
]);

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

export async function mockLoadWorkflow(input: { workflowId: string }): Promise<unknown> {
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
  await delay(120);
  // Mirrors crates/nodes builtins — same type ids, categories, socket kinds
  // and param kinds, plus options/min/max/step matching the constraint data
  // the real backend DTO now serializes.
  return MOCK_NODE_DEFS.map((def) => NodeDefSchema.parse(def));
}

/* ───── Mock node catalog (mirror of the Rust builtins) ───── */

const SAMPLERS = ["euler", "euler a", "dpm++ 2M", "dpm++ SDE", "ddim"];
const SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform"];

const MOCK_NODE_DEFS: NodeDef[] = [
  {
    type: "builtin.string",
    displayName: "String",
    category: "Input",
    inputs: [],
    outputs: [{ id: "value", kind: "string", label: "value" }],
    parameters: [{ id: "value", label: "Value", kind: "string", default: "" }],
  },
  {
    type: "builtin.load_image",
    displayName: "Load Image",
    category: "Input",
    inputs: [],
    outputs: [{ id: "image", kind: "image", label: "image" }],
    parameters: [{ id: "image", label: "Image path", kind: "string", default: "" }],
  },
  {
    type: "builtin.checkpoint_loader",
    displayName: "Checkpoint Loader",
    category: "Model",
    inputs: [],
    outputs: [
      { id: "model", kind: "model", label: "model" },
      { id: "clip", kind: "clip", label: "clip" },
      { id: "vae", kind: "vae", label: "vae" },
    ],
    parameters: [
      {
        id: "checkpoint",
        label: "Checkpoint",
        kind: "select",
        default: "sdxl_base_1.0.safetensors",
        options: [
          "sdxl_base_1.0.safetensors",
          "dreamshaper_8.safetensors",
          "rev_animated.safetensors",
        ],
      },
    ],
  },
  {
    type: "builtin.clip_text_encode",
    displayName: "CLIP Text Encode",
    category: "Conditioning",
    inputs: [{ id: "clip", kind: "clip", label: "clip" }],
    outputs: [{ id: "conditioning", kind: "conditioning", label: "conditioning" }],
    parameters: [{ id: "text", label: "Text", kind: "text", default: "" }],
  },
  {
    type: "builtin.empty_latent_image",
    displayName: "Empty Latent Image",
    category: "Latent",
    inputs: [],
    outputs: [{ id: "latent", kind: "latent", label: "latent" }],
    parameters: [
      { id: "width", label: "Width", kind: "int", default: 1024 },
      { id: "height", label: "Height", kind: "int", default: 1024 },
      { id: "batch_size", label: "Batch size", kind: "int", default: 1 },
    ],
  },
  {
    type: "builtin.ksampler",
    displayName: "KSampler",
    category: "Sampling",
    inputs: [
      { id: "model", kind: "model", label: "model" },
      { id: "positive", kind: "conditioning", label: "positive" },
      { id: "negative", kind: "conditioning", label: "negative" },
      { id: "latent", kind: "latent", label: "latent" },
    ],
    outputs: [{ id: "latent", kind: "latent", label: "latent" }],
    parameters: [
      { id: "seed", label: "Seed", kind: "int", default: 12345 },
      { id: "steps", label: "Steps", kind: "int", default: 30, min: 1, max: 100 },
      { id: "cfg", label: "CFG scale", kind: "float", default: 8.0, min: 1, max: 20 },
      { id: "sampler", label: "Sampler", kind: "select", default: "dpm++ 2M", options: SAMPLERS },
      {
        id: "scheduler",
        label: "Scheduler",
        kind: "select",
        default: "karras",
        options: SCHEDULERS,
      },
      { id: "denoise", label: "Denoise", kind: "float", default: 0.5, min: 0, max: 1 },
    ],
  },
  {
    type: "builtin.vae_decode",
    displayName: "VAE Decode",
    category: "Image",
    inputs: [
      { id: "vae", kind: "vae", label: "vae" },
      { id: "latent", kind: "latent", label: "latent" },
    ],
    outputs: [{ id: "image", kind: "image", label: "image" }],
    parameters: [],
  },
  {
    type: "builtin.vae_encode",
    displayName: "VAE Encode",
    category: "Latent",
    inputs: [
      { id: "vae", kind: "vae", label: "vae" },
      { id: "image", kind: "image", label: "image" },
    ],
    outputs: [{ id: "latent", kind: "latent", label: "latent" }],
    parameters: [],
  },
  {
    type: "builtin.save_image",
    displayName: "Save Image",
    category: "Image",
    inputs: [{ id: "image", kind: "image", label: "image" }],
    outputs: [],
    parameters: [
      { id: "filename_prefix", label: "Filename prefix", kind: "string", default: "reimagine" },
    ],
  },
  {
    type: "builtin.preview_image",
    displayName: "Preview Image",
    category: "Image",
    inputs: [{ id: "image", kind: "image", label: "image" }],
    outputs: [],
    parameters: [],
  },
];

export async function mockResolveArtifact(artifactId: string): Promise<ArtifactMetadata> {
  await delay(100);
  return {
    id: artifactId,
    nodeId: "node-save-image",
    mediaType: "image/svg+xml",
    filename: `${artifactId}.svg`,
    // Data URL so the mock preview renders in a plain browser.
    path: mockArtifactDataUrl(artifactId),
  };
}

/** Deterministic gradient placeholder image for mock artifacts. */
function mockArtifactDataUrl(artifactId: string): string {
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256">` +
    `<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">` +
    `<stop offset="0" stop-color="#7928ca"/><stop offset="1" stop-color="#50e3c2"/>` +
    `</linearGradient></defs>` +
    `<rect width="256" height="256" fill="url(#g)"/>` +
    `<text x="128" y="128" font-family="system-ui, sans-serif" font-size="13" fill="#ffffff" text-anchor="middle" dominant-baseline="middle">${artifactId}</text>` +
    `</svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
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

export async function mockSearchModels(input: {
  query: string;
  filters?: unknown;
}): Promise<ModelCatalogEntry[]> {
  await delay(200);
  const needle = input.query.trim().toLowerCase();
  if (!needle) return [];
  return MOCK_CATALOG.filter((entry) => entry.id.toLowerCase().includes(needle));
}

export async function mockGetModelCard(_input: { repoId: string }): Promise<ModelCard> {
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

/* ───── Worker switching mocks (BE-32) ───── */

const MOCK_SWITCH_TARGETS: WorkerSwitchTarget[] = [
  {
    installationId: "mock-burn-worker",
    version: "0.1.0",
    backendInstanceId: "burn:wgpu:default",
    backendKind: "burn",
    target: "aarch64-apple-darwin",
    installedAt: new Date().toISOString(),
    installPath: "/mock/workers/installed/mock-burn-worker",
    manifestDigest: "sha256-mock",
  },
];

export async function mockDrainAndSwitchWorker(
  input: WorkerSwitchArgs,
): Promise<WorkerSwitchResult> {
  await delay(120);
  return {
    instance: input.target,
    incarnationId: `incarnation-${Date.now()}`,
  };
}

export async function mockCancelAndSwitchWorker(
  input: WorkerSwitchArgs,
): Promise<WorkerSwitchResult> {
  await delay(120);
  return {
    instance: input.target,
    incarnationId: `incarnation-${Date.now()}`,
  };
}

export async function mockListWorkerSwitchTargets(): Promise<WorkerSwitchTarget[]> {
  await delay(30);
  return [...MOCK_SWITCH_TARGETS];
}

export async function mockRebootBackend(input: RebootBackendArgs): Promise<ComputeProfile> {
  await delay(400);
  const backend = input.selection;
  return {
    backend_profiles: [
      {
        backend,
        plugin: null,
        extension: null,
        instances: [],
        diagnostics: [],
      },
    ],
    diagnostics: [],
    topology_workers: [],
  };
}
