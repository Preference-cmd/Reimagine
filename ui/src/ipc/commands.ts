import {
  type ArtifactMetadata,
  type BoardCommandResult,
  type BoardSnapshot,
  type CommandError,
  type ComputeProfile,
  type DocumentChangedEvent,
  type Project,
  type ProjectMetadataInput,
  type DownloadEventPayload,
  type DownloadHuggingfaceModelArgs,
  type ModelCard,
  type ModelCatalogEntry,
  type ModelDownloadOutput,
  type ModelFilters,
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
  CommandErrorSchema,
  DocumentChangedEventSchema,
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
  mockSubscribeDocumentEvents,
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
  onEvent?: (event: RunEventPayload) => void,
): Promise<RunWorkflowResponse> {
  return invokeWithFallback(
    async () => {
      const { Channel, invoke } = await import("@tauri-apps/api/core");
      const channel = new Channel<RunEventPayload>();
      if (onEvent) {
        channel.onmessage = onEvent;
      }
      return invoke<RunWorkflowResponse>("run_workflow", { projectId, workflow, channel });
    },
    () => mockRunWorkflow(workflow, onEvent),
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

export function subscribeDocumentEvents(onEvent: (event: DocumentChangedEvent) => void): Promise<void> {
  return invokeWithFallback(
    async () => {
      const { Channel, invoke } = await import("@tauri-apps/api/core");
      const channel = new Channel<DocumentChangedEvent>();
      channel.onmessage = (payload) => {
        const event = DocumentChangedEventSchema.safeParse(payload);
        if (event.success) onEvent(event.data);
      };
      return invoke<void>("subscribe_document_events", { channel });
    },
    () => mockSubscribeDocumentEvents(onEvent),
  );
}

export function getBoardSnapshot(projectId: string): Promise<BoardSnapshot> {
  return dispatch("board_snapshot", null, { projectId }, mockGetBoardSnapshot);
}

export function previewBoardCommands(projectId: string, commandBatch: unknown): Promise<BoardCommandResult> {
  return dispatch("preview_board_commands", null, { projectId, commandBatch }, mockPreviewBoardCommands);
}

export function applyBoardCommands(projectId: string, commandBatch: unknown): Promise<BoardCommandResult> {
  return dispatch("apply_board_commands", null, { projectId, commandBatch }, mockApplyBoardCommands);
}

export function undoBoard(projectId: string): Promise<BoardCommandResult | null> {
  return dispatch("undo_board", null, { projectId }, mockUndoBoard);
}

export function redoBoard(projectId: string): Promise<BoardCommandResult | null> {
  return dispatch("redo_board", null, { projectId }, mockRedoBoard);
}

/* ───── Workflow persistence (F1-1/F1-2) ───── */

/** Save a workflow to the workspace `workflows/` dir; resolves to the file path. */
export function saveWorkflow(projectId: string, workflowId: string, workflowJson: unknown): Promise<string> {
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

/* ───── Agent (BE-19 streaming) ───── */

/** Create a new agent session in the given mode with the specified provider. */
export function createAgentSession(
  mode: string,
  provider: string,
): Promise<{ sessionId: string; mode: string; provider: string; startedAt: string }> {
  return invokeWithFallback(
    async () => {
      const { invoke } = await import("@tauri-apps/api/core");
      return invoke("create_agent_session", { mode, provider });
    },
    async () => ({
      sessionId: crypto.randomUUID(),
      mode,
      provider,
      startedAt: new Date().toISOString(),
    }),
  );
}

/**
 * One streaming agent event (AR-11 contract).
 *
 * kind semantics:
 *   - terminal milestones: "turn_completed" (normal) | "error" (failed)
 *   - in-flight: "content_delta", "reasoning_delta", "tool_invoked",
 *     "tool_completed", "tool_failed", "session_started",
 *     "session_stopped", "context_compacted"
 *   - compatibility: "provider_error" groups under error semantics;
 *     consumers should treat kind error | provider_error | tool_failed
 *     as failures and stop the thread.

 * A turn always ends with exactly one terminal marker.
 */
export interface AgentEventPayload {
  sessionId: string;
  kind: string;
  toolName?: string;
  toolCallId?: string;
  code?: string;
  message?: string;
  /** Turn-level observability, populated on turn_completed. */
  durationMs?: number;
  estimatedCost?: number;
  usage?: {
    inputTokens?: number;
    outputTokens?: number;
    reasoningTokens?: number;
    cacheCreationInputTokens?: number;
    cacheReadInputTokens?: number;
  };
}

/**
 * Execute an agent turn with live streaming events.
 * `onEvent` is called for each `AgentEvent` (including `content_delta`)
 * as the model generates tokens.
 */
export async function agentTurn(
  sessionId: string,
  turnId: string,
  model: string,
  input: unknown[],
  onEvent?: (event: AgentEventPayload) => void,
  outputSchema?: unknown,
  context?: unknown,
): Promise<unknown> {
  return invokeWithFallback(
    async () => {
      const { Channel, invoke } = await import("@tauri-apps/api/core");
      const channel = new Channel<AgentEventPayload>();
      if (onEvent) {
        channel.onmessage = onEvent;
      }
      return invoke("agent_turn", {
        sessionId,
        turnId,
        model,
        input,
        outputSchema: outputSchema ?? null,
        context: context ?? null,
        channel,
      });
    },
    async () => {
      // Mock: simulate a short streaming response
      if (onEvent) {
        onEvent({ sessionId, kind: "content_delta", message: "Hello! " });
        onEvent({ sessionId, kind: "content_delta", message: "This is a mock response." });
      }
      return { status: "completed", stopReason: "final_response" };
    },
  );
}

/** List available agent providers for the UI selector. */
export function listAgentProviders(): Promise<string[]> {
  return invokeWithFallback(
    async () => {
      const { invoke } = await import("@tauri-apps/api/core");
      return invoke<string[]>("list_agent_providers");
    },
    async () => ["openai", "anthropic"],
  );
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
 *  (a backend kind, `"burn"` | `"candle"`). Resolves to the new compute
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
