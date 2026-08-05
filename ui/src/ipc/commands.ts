import {
  type ArtifactMetadata,
  type DownloadEventPayload,
  type DownloadHuggingfaceModelArgs,
  type ModelCard,
  type ModelCatalogEntry,
  type ModelDownloadOutput,
  type ModelFilters,
  type ModelInfo,
  type NodeDef,
  type RunEventPayload,
  type RunWorkflowResponse,
  type Workflow,
  DownloadHuggingfaceModelArgsSchema,
} from "./schemas";
import {
  mockCancelRun,
  mockDownloadHuggingfaceModel,
  mockGetModelCard,
  mockGetNodeDefs,
  mockListModels,
  mockOpenArtifact,
  mockResolveArtifact,
  mockRunWorkflow,
  mockSearchModels,
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

export async function resolveArtifact(
  artifactId: string,
): Promise<ArtifactMetadata> {
  if (USE_MOCK) {
    return mockResolveArtifact(artifactId);
  }
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<ArtifactMetadata>("resolve_artifact", { artifactId });
}

export async function openArtifact(artifactId: string): Promise<void> {
  if (USE_MOCK) {
    return mockOpenArtifact(artifactId);
  }
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<void>("open_artifact", { artifactId });
}

export function searchModels(
  query: string,
  filters?: ModelFilters,
): Promise<ModelCatalogEntry[]> {
  return dispatch(
    "search_models",
    null,
    { query, filters },
    mockSearchModels,
  );
}

export function getModelCard(repoId: string): Promise<ModelCard> {
  return dispatch("get_model_card", null, { repoId }, mockGetModelCard);
}

export async function downloadHuggingfaceModel(
  args: DownloadHuggingfaceModelArgs,
  onEvent?: (event: DownloadEventPayload) => void,
): Promise<ModelDownloadOutput> {
  DownloadHuggingfaceModelArgsSchema.parse(args);
  if (USE_MOCK) {
    return mockDownloadHuggingfaceModel(args, onEvent);
  }
  const { Channel, invoke } = await import("@tauri-apps/api/core");
  const channel = new Channel<DownloadEventPayload>();
  if (onEvent) {
    channel.onmessage = onEvent;
  }
  return invoke<ModelDownloadOutput>("download_huggingface_model", {
    repoId: args.repoId,
    revision: args.revision,
    allowPatterns: args.allowPatterns,
    targetRelativeDir: args.targetRelativeDir,
    overwrite: args.overwrite,
    autoDetect: args.autoDetect,
    fromCatalog: args.fromCatalog,
    channel,
  });
}
