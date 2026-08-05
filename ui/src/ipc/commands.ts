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
  type WorkflowFileSummary,
  DownloadHuggingfaceModelArgsSchema,
} from "./schemas";
import {
  mockCancelRun,
  mockDownloadHuggingfaceModel,
  mockGetModelCard,
  mockGetNodeDefs,
  mockListModels,
  mockListWorkflows,
  mockLoadWorkflow,
  mockOpenArtifact,
  mockResolveArtifact,
  mockRunWorkflow,
  mockSaveWorkflow,
  mockSearchModels,
} from "./mock";

/**
 * Mock polarity (F1-4): real IPC is always attempted first; mocks are used
 * only when the Tauri bridge is unavailable AND mocks are enabled.
 *   - `VITE_FORCE_MOCK=1` is an explicit override that skips IPC entirely.
 *   - In dev builds mocks are enabled; in production they are not.
 */
const FORCE_MOCK = import.meta.env.VITE_FORCE_MOCK === "1";
const MOCKS_ENABLED = FORCE_MOCK || import.meta.env.DEV;

/** True when the current webview exposes the Tauri IPC bridge. */
function tauriAvailable(): boolean {
  try {
    return (
      typeof window !== "undefined" &&
      typeof (window as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__ !==
        "undefined"
    );
  } catch {
    return false;
  }
}

/** Best-effort detection of "Tauri isn't here" errors from @tauri-apps/api. */
function isTauriUnavailable(error: unknown): boolean {
  if (!error || typeof error !== "object") return false;
  const message = String(
    (error as { message?: unknown }).message ?? error,
  );
  return (
    message.includes("__TAURI_INTERNALS__") ||
    message.includes("Tauri") ||
    message.includes("is not a function")
  );
}

/** Real IPC first; mock fallback only when Tauri is unavailable and mocks are on. */
async function dispatch<TIn, TOut>(
  name: string,
  schema: { parse: (x: unknown) => TIn } | null,
  input: TIn,
  mockFn: (i: TIn) => Promise<TOut>,
): Promise<TOut> {
  if (schema) schema.parse(input);
  if (FORCE_MOCK) return mockFn(input);
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    // Spread the input so its keys map to the command's named arguments
    // (e.g. `search_models(query, filters)`). `undefined` input (no-arg
    // commands) spreads to an empty args object.
    const args = { ...(input as object) };
    return await invoke<TOut>(name, args);
  } catch (e) {
    if (MOCKS_ENABLED && (isTauriUnavailable(e) || !tauriAvailable())) {
      return mockFn(input);
    }
    throw e;
  }
}

/** Shared fallback for commands that need a Channel or custom invoke shape. */
async function invokeWithFallback<T>(
  real: () => Promise<T>,
  mock: () => Promise<T>,
): Promise<T> {
  if (FORCE_MOCK) return mock();
  try {
    return await real();
  } catch (e) {
    if (MOCKS_ENABLED && (isTauriUnavailable(e) || !tauriAvailable())) {
      return mock();
    }
    throw e;
  }
}

export async function runWorkflow(
  workflow: Workflow,
  onEvent?: (event: RunEventPayload) => void,
): Promise<RunWorkflowResponse> {
  return invokeWithFallback(
    async () => {
      const { Channel, invoke } = await import("@tauri-apps/api/core");
      const channel = new Channel<RunEventPayload>();
      if (onEvent) {
        channel.onmessage = onEvent;
      }
      return invoke<RunWorkflowResponse>("run_workflow", { workflow, channel });
    },
    () => mockRunWorkflow(workflow),
  );
}

export async function cancelRun(runId: string): Promise<void> {
  return invokeWithFallback(
    async () => {
      const { invoke } = await import("@tauri-apps/api/core");
      return invoke<void>("cancel_run", { runId });
    },
    () => mockCancelRun(runId),
  );
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
  return invokeWithFallback(
    async () => {
      const { invoke } = await import("@tauri-apps/api/core");
      return invoke<ArtifactMetadata>("resolve_artifact", { artifactId });
    },
    () => mockResolveArtifact(artifactId),
  );
}

export async function openArtifact(artifactId: string): Promise<void> {
  return invokeWithFallback(
    async () => {
      const { invoke } = await import("@tauri-apps/api/core");
      return invoke<void>("open_artifact", { artifactId });
    },
    () => mockOpenArtifact(artifactId),
  );
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
  return invokeWithFallback(
    async () => {
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
    },
    () => mockDownloadHuggingfaceModel(args, onEvent),
  );
}

/* ───── Workflow persistence (F1-1/F1-2) ───── */

/** Save a workflow to the workspace `workflows/` dir; resolves to the file path. */
export function saveWorkflow(
  workflowId: string,
  workflowJson: unknown,
): Promise<string> {
  return dispatch(
    "save_workflow",
    null,
    { workflowId, workflowJson },
    mockSaveWorkflow,
  );
}

/** Load a saved workflow (backend `Workflow` JSON) by id. */
export function loadWorkflow(workflowId: string): Promise<unknown> {
  return dispatch(
    "load_workflow",
    null,
    { workflowId },
    mockLoadWorkflow,
  );
}

/** List saved workflows (newest first). */
export function listWorkflows(): Promise<WorkflowFileSummary[]> {
  return dispatch(
    "list_workflows",
    null,
    undefined,
    mockListWorkflows,
  );
}
