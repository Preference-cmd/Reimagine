import {
  WorkflowSchema,
  RunIdSchema,
  type Workflow,
  type RunId,
  type ModelInfo,
  type NodeDef,
} from "./schemas";

/* ───── Helpers ───── */

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

function rand(prefix: string) {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

/* ───── Mock data ───── */

const MOCK_NODES: NodeDef[] = [
  {
    type: "clipEncode",
    displayName: "CLIP Text Encode (Prompt)",
    category: "conditioning",
    inputs: [],
    outputs: [{ id: "cond", kind: "conditioning", label: "CONDITIONING" }],
    parameters: [
      { id: "text", label: "text", kind: "text", default: "" },
    ],
  },
  {
    type: "ksampler",
    displayName: "KSampler",
    category: "sampling",
    inputs: [
      { id: "model", kind: "model", label: "model" },
      { id: "positive", kind: "conditioning", label: "positive" },
      { id: "negative", kind: "conditioning", label: "negative" },
      { id: "latent_image", kind: "latent", label: "latent_image" },
    ],
    outputs: [{ id: "latent", kind: "latent", label: "LATENT" }],
    parameters: [
      { id: "seed", label: "seed", kind: "int", default: 0 },
      { id: "steps", label: "steps", kind: "int", default: 30, min: 1, max: 200 },
      { id: "cfg", label: "cfg", kind: "float", default: 7.0, min: 0, max: 30 },
    ],
  },
  {
    type: "emptyLatent",
    displayName: "Empty Latent Image",
    category: "latent",
    inputs: [],
    outputs: [{ id: "latent", kind: "latent", label: "LATENT" }],
    parameters: [
      { id: "width", label: "width", kind: "int", default: 1024, min: 64, max: 4096 },
      { id: "height", label: "height", kind: "int", default: 1024, min: 64, max: 4096 },
    ],
  },
  {
    type: "vaeDecode",
    displayName: "VAE Decode",
    category: "vae",
    inputs: [
      { id: "samples", kind: "latent", label: "samples" },
      { id: "vae", kind: "model", label: "vae" },
    ],
    outputs: [{ id: "image", kind: "image", label: "IMAGE" }],
    parameters: [],
  },
  {
    type: "saveImage",
    displayName: "Save Image",
    category: "output",
    inputs: [{ id: "images", kind: "image", label: "images" }],
    outputs: [],
    parameters: [
      { id: "filename_prefix", label: "filename_prefix", kind: "string", default: "reimagine" },
    ],
  },
];

const MOCK_MODELS: ModelInfo[] = [
  {
    id: "sd_xl_base_1.0",
    name: "SDXL Base 1.0",
    family: "stable-diffusion-xl",
    size: "6.94 GB",
    path: "/models/sd_xl_base_1.0.safetensors",
  },
  {
    id: "dreamshaper_8",
    name: "DreamShaper 8",
    family: "stable-diffusion-1.5",
    size: "2.07 GB",
    path: "/models/dreamshaper_8.safetensors",
  },
];

/* ───── Mock command implementations ───── */

export async function mockRunWorkflow(workflow: Workflow): Promise<RunId> {
  await delay(200);
  WorkflowSchema.parse(workflow); // validates; result unused but ensures shape
  return RunIdSchema.parse(rand("run"));
}

export async function mockCancelRun(_runId: RunId): Promise<void> {
  await delay(100);
}

export async function mockListModels(): Promise<ModelInfo[]> {
  await delay(150);
  return [...MOCK_MODELS];
}

export async function mockGetNodeDefs(): Promise<NodeDef[]> {
  await delay(100);
  return [...MOCK_NODES];
}