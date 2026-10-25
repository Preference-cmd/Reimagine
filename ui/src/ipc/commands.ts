import {
  type ArtifactMetadata,
  type BoardCommandResultDto,
  type BoardSnapshot,
  type CommandError,
  type ComputeProfile,
  type Event,
  type Project,
  type ProjectMetadataInput,
  type DownloadHuggingfaceModelArgs,
  type ModelCard,
  type ModelCatalogEntry,
  type ModelDownloadOutput,
  type ModelFilters,
  type ModelInfo,
  type NodeDef,
  type RebootBackendArgs,
  type RunWorkflowResponse,
  type Workflow,
  type WorkflowFileSummary,
  type WorkerSwitchArgs,
  type WorkerSwitchResult,
  type WorkerSwitchTarget,
  CommandErrorSchema,
  EventSchema,
  RunWorkflowResponseSchema,
  ProjectMetadataInputSchema,
  DownloadHuggingfaceModelArgsSchema,
  RebootBackendArgsSchema,
  WorkerSwitchArgsSchema,
} from "./schemas";
import {
  mockCancelAndSwitchWorker,
  mockCancelRun,
  mockCreateProject,
  mockDeleteProject,
  mockGetBoardSnapshot,
  mockListProjects,
  mockLoadProject,
  mockSetActiveProject,
  mockSubscribeEvents,
  mockUndoBoard,
  mockRedoBoard,
  mockApplyBoardCommands,
  mockPreviewBoardCommands,
  mockUpdateProject,
  mockDownloadHuggingfaceModel,
  mockDrainAndSwitchWorker,
  mockGetModelCard,
  mockGetNodeDefs,
  mockListModels,
  mockListWorkerSwitchTargets,
  mockListWorkflows,
  mockLoadWorkflow,
  mockOpenArtifact,
  mockRebootBackend,
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
      typeof (window as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__ !== "undefined"
    );
  } catch {
    return false;
  }
}

/** Best-effort detection of "Tauri isn't here" errors from @tauri-apps/api. */
function isTauriUnavailable(error: unknown): boolean {
  if (!error || typeof error !== "object") return false;
  const message = String((error as { message?: unknown }).message ?? error);
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
async function invokeWithFallback<T>(real: () => Promise<T>, mock: () => Promise<T>): Promise<T> {
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
  projectId: string,
  workflow: Workflow,
): Promise<RunWorkflowResponse> {
  const response = await dispatch("run_workflow", null, { projectId, workflow }, () =>
    mockRunWorkflow(workflow),
  );
  // Parse so a host/UI field mismatch fails here instead of leaving `runId`
  // undefined and the run store ignoring every run event.
  return RunWorkflowResponseSchema.parse(response);
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

export async function resolveArtifact(artifactId: string): Promise<ArtifactMetadata> {
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

export function searchModels(query: string, filters?: ModelFilters): Promise<ModelCatalogEntry[]> {
  return dispatch("search_models", null, { query, filters }, mockSearchModels);
}

export function getModelCard(repoId: string): Promise<ModelCard> {
  return dispatch("get_model_card", null, { repoId }, mockGetModelCard);
}

export async function downloadHuggingfaceModel(
  args: DownloadHuggingfaceModelArgs,
): Promise<ModelDownloadOutput> {
  return dispatch(
    "download_huggingface_model",
    DownloadHuggingfaceModelArgsSchema,
    args,
    mockDownloadHuggingfaceModel,
  );
}

/* ───── Project / Board surface (AR-39) ───── */

export function listProjects(): Promise<Project[]> {
  return dispatch("list_projects", null, undefined, mockListProjects);
}

export function createProject(projectId: string, metadata: ProjectMetadataInput): Promise<Project> {
  ProjectMetadataInputSchema.parse(metadata);
  return dispatch("create_project", null, { projectId, metadata }, mockCreateProject);
}

export function loadProject(projectId: string): Promise<Project> {
  return dispatch("load_project", null, { projectId }, mockLoadProject);
}

export function updateProject(projectId: string, metadata: ProjectMetadataInput): Promise<Project> {
  ProjectMetadataInputSchema.parse(metadata);
  return dispatch("update_project", null, { projectId, metadata }, mockUpdateProject);
}

export function deleteProject(projectId: string): Promise<void> {
  return dispatch("delete_project", null, { projectId }, mockDeleteProject);
}

export function setActiveProject(projectId: string): Promise<Project> {
  return dispatch("set_active_project", null, { projectId }, mockSetActiveProject);
}

/* ───── Events: one host subscription fanned out to listeners ───── */

type EventListener = (event: Event) => void;
const eventListeners = new Set<EventListener>();
let eventSubscriptionPromise: Promise<void> | null = null;

function ensureEventSubscription(): Promise<void> {
  if (!eventSubscriptionPromise) {
    eventSubscriptionPromise = invokeWithFallback(
      async () => {
        const { Channel, invoke } = await import("@tauri-apps/api/core");
        const channel = new Channel<Event>();
        channel.onmessage = (payload) => {
          const parsed = EventSchema.safeParse(payload);
          if (parsed.success) {
            for (const listener of Array.from(eventListeners)) {
              listener(parsed.data);
            }
          }
        };
        return invoke<void>("subscribe_events", { channel });
      },
      () =>
        mockSubscribeEvents((event) => {
          for (const listener of Array.from(eventListeners)) {
            listener(event);
          }
        }),
    );
  }
  return eventSubscriptionPromise;
}

export function subscribeEvents(listener: EventListener): () => void {
  eventListeners.add(listener);
  void ensureEventSubscription();
  return () => {
    eventListeners.delete(listener);
  };
}

export function getBoardSnapshot(projectId: string): Promise<BoardSnapshot> {
  return dispatch("board_snapshot", null, { projectId }, mockGetBoardSnapshot);
}

export function previewBoardCommands(
  projectId: string,
  commandBatch: unknown,
): Promise<BoardCommandResultDto> {
  return dispatch(
    "preview_board_commands",
    null,
    { projectId, commandBatch },
    mockPreviewBoardCommands,
  );
}

export function applyBoardCommands(
  projectId: string,
  commandBatch: unknown,
): Promise<BoardCommandResultDto> {
  return dispatch(
    "apply_board_commands",
    null,
    { projectId, commandBatch },
    mockApplyBoardCommands,
  );
}

export function undoBoard(projectId: string): Promise<BoardCommandResultDto | null> {
  return dispatch("undo_board", null, { projectId }, mockUndoBoard);
}

export function redoBoard(projectId: string): Promise<BoardCommandResultDto | null> {
  return dispatch("redo_board", null, { projectId }, mockRedoBoard);
}

/* ───── Workflow persistence (F1-1/F1-2) ───── */

/** Save a workflow to the workspace `workflows/` dir; resolves to the file path. */
export function saveWorkflow(
  projectId: string,
  workflowId: string,
  workflowJson: unknown,
): Promise<string> {
  return dispatch("save_workflow", null, { projectId, workflowId, workflowJson }, mockSaveWorkflow);
}

/** Load a saved workflow (backend `Workflow` JSON) by id. */
export function loadWorkflow(projectId: string, workflowId: string): Promise<unknown> {
  return dispatch("load_workflow", null, { projectId, workflowId }, mockLoadWorkflow);
}

/** List saved workflows (newest first). */
export function listWorkflows(projectId: string): Promise<WorkflowFileSummary[]> {
  return dispatch("list_workflows", null, { projectId }, mockListWorkflows);
}

/* ───── Worker switching (BE-32) ───── */

/** Drain in-flight runs (waiting up to `deadlineSecs`) then switch the active
 *  worker to the installed worker for `args.target` (a backend instance id). */
export function drainAndSwitchWorker(args: WorkerSwitchArgs): Promise<WorkerSwitchResult> {
  return dispatch(
    "drain_and_switch_worker",
    WorkerSwitchArgsSchema,
    args,
    mockDrainAndSwitchWorker,
  );
}

/** Cancel in-flight runs, then switch the active worker to `args.target`. */
export function cancelAndSwitchWorker(args: WorkerSwitchArgs): Promise<WorkerSwitchResult> {
  return dispatch(
    "cancel_and_switch_worker",
    WorkerSwitchArgsSchema,
    args,
    mockCancelAndSwitchWorker,
  );
}

/** List installed workers usable as switch targets. */
export function listWorkerSwitchTargets(): Promise<WorkerSwitchTarget[]> {
  return dispatch("list_worker_switch_targets", null, undefined, mockListWorkerSwitchTargets);
}

/** Drain in-flight runs and re-bootstrap the workspace with `selection`
 *  (a backend kind, `"burn"`). Resolves to the new compute
 *  profile. The selection is not persisted across app restarts. */
export function rebootBackend(args: RebootBackendArgs): Promise<ComputeProfile> {
  return dispatch("rebootstrap_backend", RebootBackendArgsSchema, args, mockRebootBackend);
}

/**
 * Normalize a thrown IPC error into the structured BE-31 shape.
 *
 * Returns null when the error is not a structured command error (e.g. a
 * transport-level failure); callers can then treat it as a generic failure.
 */
export function toCommandError(error: unknown): CommandError | null {
  if (!error || typeof error !== "object") return null;
  const parsed = CommandErrorSchema.safeParse(error);
  return parsed.success ? parsed.data : null;
}
