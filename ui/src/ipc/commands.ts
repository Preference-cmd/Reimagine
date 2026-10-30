import {
  type BoardCommandResultDto,
  type BoardSnapshot,
  type CommandError,
  type Event,
  type Project,
  type ProjectMetadataInput,
  type DownloadHuggingfaceModelArgs,
  type ModelCard,
  type ModelCatalogEntry,
  type ModelDownloadOutput,
  type ModelFilters,
  type ModelInfo,
  type InvokeResult,
  CommandErrorSchema,
  EventSchema,
  ProjectMetadataInputSchema,
  DownloadHuggingfaceModelArgsSchema,
  BoardSnapshotSchema,
} from "./schemas";
import {
  mockCreateProject,
  mockGetBoardSnapshot,
  mockListProjects,
  mockLoadProject,
  mockSetActiveProject,
  mockSubscribeEvents,
  mockUndoBoard,
  mockRedoBoard,
  mockApplyBoardCommands,
  mockPreviewBoardCommands,
  mockDownloadHuggingfaceModel,
  mockGetModelCard,
  mockListModels,
  mockSearchModels,
  mockOpInvoke,
  mockReadAssetPreview,
} from "./mock";
import { getAssetUrl } from "../lib/assetUrl";

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

/** Public shell check: false in a plain browser preview, true in Tauri. */
export function isTauri(): boolean {
  return tauriAvailable();
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

export function listModels(): Promise<ModelInfo[]> {
  return dispatch("list_models", null, undefined, mockListModels);
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

export async function getBoardSnapshot(projectId: string): Promise<BoardSnapshot> {
  const result = await dispatch("board_snapshot", null, { projectId }, mockGetBoardSnapshot);
  return BoardSnapshotSchema.parse(result);
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

/** Undo the top board history entry, or reject when `expectedEntryId` is not
 *  the current top. Passing null keeps the unguarded host behavior. */
export function undoBoard(
  projectId: string,
  expectedEntryId?: string | null,
): Promise<BoardCommandResultDto | null> {
  return dispatch(
    "undo_board",
    null,
    { projectId, expectedEntryId: expectedEntryId ?? null },
    mockUndoBoard,
  );
}

/** Redo the top undone board history entry, or reject when `expectedEntryId`
 *  is not the current top. Passing null keeps the unguarded host behavior. */
export function redoBoard(
  projectId: string,
  expectedEntryId?: string | null,
): Promise<BoardCommandResultDto | null> {
  return dispatch(
    "redo_board",
    null,
    { projectId, expectedEntryId: expectedEntryId ?? null },
    mockRedoBoard,
  );
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

/* ───── Operation Orchestration (W-a / W-b) ───── */

export function opInvoke(projectId: string, name: string, input: unknown): Promise<InvokeResult> {
  return dispatch("op_invoke", null, { projectId, name, input }, () =>
    mockOpInvoke({ projectId, name, input }),
  );
}

/** Media type of an asset and, for textual assets, its first `maxLines` lines. */
export type AssetPreview = { url: string; mediaType: string; text: string | null };

function isTextual(mediaType: string): boolean {
  return mediaType.startsWith("text/") || mediaType === "application/json";
}

/** Read an asset through the `reimagine-asset` protocol; HTTP errors reject. */
export async function readAssetPreview(
  projectId: string,
  assetId: string,
  maxLines: number,
): Promise<AssetPreview> {
  const url = getAssetUrl(projectId, assetId);
  return await invokeWithFallback(
    async () => {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status} reading asset ${assetId}`);
      const mediaType = res.headers.get("content-type") ?? "application/octet-stream";
      const text = isTextual(mediaType)
        ? (await res.text()).split("\n").slice(0, maxLines).join("\n")
        : null;
      return { url, mediaType, text };
    },
    () => mockReadAssetPreview(url, assetId, maxLines),
  );
}
