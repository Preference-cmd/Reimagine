import {
  type ActivityRecord,
  type ArtifactMetadata,
  type BoardCommandResultDto,
  type BoardItem,
  type BoardSnapshot,
  type ComputeProfile,
  type Event,
  type Project,
  type ProjectMetadataInput,
  type DownloadHuggingfaceModelArgs,
  type DownloadProgress,
  type ModelCard,
  type ModelCatalogEntry,
  type ModelDownloadOutput,
  type ModelInfo,
  type NodeDef,
  type RebootBackendArgs,
  type WorkflowFileSummary,
  type WorkerSwitchArgs,
  type WorkerSwitchResult,
  type WorkerSwitchTarget,
  NodeDefSchema,
  type InvokeResult,
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

/* ───── Workflow persistence mocks (localStorage-backed) ─────
   Kept in localStorage so dev sessions survive reloads, mirroring the
   on-disk `workflows/` behavior of the real backend. */

const PERSISTENCE_KEY = "reimagine.mock.workflows";
const PROJECTS_KEY = "reimagine.mock.projects";
const eventSubscribers = new Set<(event: Event) => void>();

function emitEvent(event: Event): void {
  for (const subscriber of eventSubscribers) subscriber(event);
}

function mockReadAll(): Record<string, string> {
  try {
    return JSON.parse(localStorage.getItem(PERSISTENCE_KEY) ?? "{}");
  } catch {
    return {};
  }
}

function mockReadProjects(): Record<string, Project> {
  const fallback: Project = {
    id: "default",
    name: "Default",
    description: "Default workspace project",
    createdAt: "2026-08-18T00:00:00Z",
    updatedAt: "2026-08-18T00:00:00Z",
  };
  try {
    const parsed = JSON.parse(localStorage.getItem(PROJECTS_KEY) ?? "{}");
    return Object.keys(parsed).length > 0 ? parsed : { default: fallback };
  } catch {
    return { default: fallback };
  }
}

function mockWriteProjects(projects: Record<string, Project>): void {
  localStorage.setItem(PROJECTS_KEY, JSON.stringify(projects));
}

function mockWriteAll(all: Record<string, string>): void {
  localStorage.setItem(PERSISTENCE_KEY, JSON.stringify(all));
}

export async function mockSaveWorkflow(input: {
  projectId: string;
  workflowId: string;
  workflowJson: unknown;
}): Promise<string> {
  await delay(50);
  const all = mockReadAll();
  const key = `${input.projectId}/${input.workflowId}`;
  all[key] = JSON.stringify(input.workflowJson);
  emitEvent({
    type: "document",
    kind: "workflow.changed",
    projectId: input.projectId,
    documentId: input.workflowId,
    version: Number((input.workflowJson as { version?: unknown })?.version ?? 0),
  });
  mockWriteAll(all);
  return `mock:/projects/${input.projectId}/workflows/${input.workflowId}.json`;
}

export async function mockLoadWorkflow(input: {
  projectId: string;
  workflowId: string;
}): Promise<unknown> {
  await delay(50);
  const key = `${input.projectId}/${input.workflowId}`;
  const raw = mockReadAll()[key];
  if (!raw) {
    throw new Error(`mock workflow not found: ${key}`);
  }
  return JSON.parse(raw);
}

export async function mockListWorkflows(input: {
  projectId: string;
}): Promise<WorkflowFileSummary[]> {
  await delay(30);
  const prefix = `${input.projectId}/`;
  const entries = Object.entries(mockReadAll()).filter(([key]) => key.startsWith(prefix));
  return entries.map(([key], index) => ({
    id: key.slice(prefix.length),
    modified_millis: entries.length - index,
  }));
}

export async function mockSubscribeEvents(onEvent: (event: Event) => void): Promise<void> {
  eventSubscribers.add(onEvent);
}

export async function mockListProjects(): Promise<Project[]> {
  await delay(30);
  return Object.values(mockReadProjects());
}

export async function mockCreateProject(input: {
  projectId: string;
  metadata: ProjectMetadataInput;
}): Promise<Project> {
  const projects = mockReadProjects();
  const now = new Date().toISOString();
  const project: Project = {
    id: input.projectId,
    name: input.metadata.name,
    description: input.metadata.description,
    createdAt: input.metadata.createdAt ?? now,
    updatedAt: input.metadata.updatedAt ?? now,
  };
  projects[input.projectId] = project;
  mockWriteProjects(projects);
  return project;
}

export async function mockLoadProject(input: { projectId: string }): Promise<Project> {
  const project = mockReadProjects()[input.projectId];
  if (!project) throw new Error("mock project not found: " + input.projectId);
  return project;
}

export async function mockUpdateProject(input: {
  projectId: string;
  metadata: ProjectMetadataInput;
}): Promise<Project> {
  const current = await mockLoadProject({ projectId: input.projectId });
  const project = {
    ...current,
    name: input.metadata.name,
    description: input.metadata.description,
    updatedAt: input.metadata.updatedAt ?? new Date().toISOString(),
  };
  const projects = mockReadProjects();
  projects[input.projectId] = project;
  mockWriteProjects(projects);
  return project;
}

export async function mockDeleteProject(input: { projectId: string }): Promise<void> {
  const projects = mockReadProjects();
  delete projects[input.projectId];
  mockWriteProjects(projects);
  mockResetBoard(input.projectId);
}

export async function mockSetActiveProject(input: { projectId: string }): Promise<Project> {
  return mockLoadProject(input);
}

/** One undoable board batch: the items before it, its history id and targets. */
type MockBoardHistoryEntry = {
  items: BoardItem[];
  entryId: string;
  targets: string[];
};

const mockBoards: Record<string, BoardSnapshot> = {};
const mockBoardHistory: Record<
  string,
  {
    past: MockBoardHistoryEntry[];
    future: MockBoardHistoryEntry[];
  }
> = {};

function mockBoardEntryId(
  history: { past: MockBoardHistoryEntry[]; future: MockBoardHistoryEntry[] } | undefined,
  direction: "past" | "future",
): string | null {
  const entries = history?.[direction];
  if (!entries || entries.length === 0) return null;
  return entries[entries.length - 1].entryId;
}

/**
 * Mirrors the host's history guard: reject before touching the board, the
 * history stacks or the activity feed. The details carry both current tops so
 * the UI can resynchronize its stacks.
 */
function rejectBoardHistoryMismatch(
  projectId: string,
  action: "undo" | "redo",
  expectedEntryId: string,
  history: { past: MockBoardHistoryEntry[]; future: MockBoardHistoryEntry[] } | undefined,
): never {
  const undoEntryId = mockBoardEntryId(history, "past");
  const redoEntryId = mockBoardEntryId(history, "future");
  const currentEntryId = action === "undo" ? undoEntryId : redoEntryId;
  throw {
    code: "conflict",
    message: `board ${action} for project \`${projectId}\` expected history entry \`${expectedEntryId}\`, but the current ${action} entry is ${currentEntryId === null ? "none" : `\`${currentEntryId}\``}`,
    details: {
      project_id: projectId,
      action,
      expected_entry_id: expectedEntryId,
      undo_entry_id: undoEntryId,
      redo_entry_id: redoEntryId,
    },
  };
}

export function mockResetBoard(projectId?: string): void {
  if (projectId) {
    delete mockBoards[projectId];
    delete mockBoardHistory[projectId];
    delete mockActivity[projectId];
  } else {
    for (const key of Object.keys(mockBoards)) {
      delete mockBoards[key];
    }
    for (const key of Object.keys(mockBoardHistory)) {
      delete mockBoardHistory[key];
    }
    for (const key of Object.keys(mockActivity)) {
      delete mockActivity[key];
    }
  }
}

/* ───── Dev activity feed (activity.list, activity.appended) ─────
   Fixture rows make the browser preview show the panel. Live board edits
   append a record and emit the same event the host sends, so the feed, the
   undo stack and the canvas trace behave as they do in Tauri. */

const mockActivity: Record<string, ActivityRecord[]> = {};
let mockActivitySequence = 0;

function mockDemoActivity(projectId: string): ActivityRecord[] {
  const now = Date.now();
  const at = (minutesAgo: number) => new Date(now - minutesAgo * 60_000).toISOString();
  return [
    {
      id: `act_demo_agent_${projectId}`,
      at: at(2),
      actor: { kind: "Agent", id: "agent-demo", label: "Agent" },
      summary: "Applied 2 board changes",
      summaryKey: "activity.board.apply",
      summaryParams: { count: 2 },
      targets: ["note-demo-2", "note-demo-3"],
      kind: "applied",
      historyEntryId: "board-history:demo-agent",
    },
    {
      id: `act_demo_note_${projectId}`,
      at: at(5),
      actor: { kind: "Human", id: null, label: null },
      summary: "Added a note",
      summaryKey: "activity.board.addNote",
      summaryParams: { count: 1 },
      targets: ["note-demo-1"],
      kind: "applied",
      historyEntryId: "board-history:demo-note",
    },
    {
      id: `act_demo_op_${projectId}`,
      at: at(12),
      actor: { kind: "Human", id: null, label: null },
      summary: "Ran collection.create",
      summaryKey: "activity.op",
      summaryParams: { op: "collection.create" },
      targets: [],
      kind: "applied",
      historyEntryId: null,
    },
    {
      id: `act_demo_undo_${projectId}`,
      at: at(20),
      actor: { kind: "Human", id: null, label: null },
      summary: "Undid the last board change",
      summaryKey: "activity.board.undo",
      summaryParams: {},
      targets: ["note-demo-1"],
      kind: "undone",
      historyEntryId: "board-history:demo-note",
    },
  ];
}

/** Activity for one project, seeded newest first on first use. */
function mockActivityFor(projectId: string): ActivityRecord[] {
  let records = mockActivity[projectId];
  if (!records) {
    records = projectId === MOCK_FIXTURE_PROJECT_ID ? mockDemoActivity(projectId) : [];
    mockActivity[projectId] = records;
  }
  return records;
}

function appendMockActivity(
  projectId: string,
  record: Omit<ActivityRecord, "id" | "at">,
): ActivityRecord {
  mockActivitySequence += 1;
  const appended: ActivityRecord = {
    ...record,
    id: `act_mock_${Date.now().toString(36)}_${mockActivitySequence}`,
    at: new Date().toISOString(),
  };
  mockActivityFor(projectId).unshift(appended);
  emitEvent({
    type: "mod",
    kind: "activity.appended",
    payload: { projectId, record: appended },
  });
  return appended;
}

/** Summary key and English fallback for an applied mock batch (host mirror). */
function mockApplySummary(commands: RawBoardCommand[]): {
  summaryKey: string;
  summary: string;
} {
  const count = commands.length;
  const allAdds = commands.every((command) => command.type === "add_item");
  const allNotes =
    allAdds &&
    commands.every((command) => command.type === "add_item" && command.kind.type === "note");

  if (allNotes) {
    return {
      summaryKey: "activity.board.addNote",
      summary: count === 1 ? "Added a note" : `Added ${count} notes`,
    };
  }
  if (allAdds) {
    return {
      summaryKey: "activity.board.addItem",
      summary: count === 1 ? "Added an item" : `Added ${count} items`,
    };
  }
  return {
    summaryKey: "activity.board.apply",
    summary: count === 1 ? "Applied a board change" : `Applied ${count} board changes`,
  };
}

export async function mockGetBoardSnapshot(input: { projectId: string }): Promise<BoardSnapshot> {
  if (!mockBoards[input.projectId]) {
    mockBoards[input.projectId] = {
      id: "board-" + input.projectId,
      projectId: input.projectId,
      version: 0,
      items: [],
    };
    mockBoardHistory[input.projectId] = { past: [], future: [] };
  }
  return mockBoards[input.projectId];
}

type RawBoardCommand =
  | {
      type: "add_item";
      item_id: string;
      kind: BoardItem["kind"];
      position: { x: number; y: number };
      size: { width: number; height: number };
    }
  | { type: "remove_item"; item_id: string }
  | { type: "move_item"; item_id: string; position: { x: number; y: number } }
  | { type: "resize_item"; item_id: string; size: { width: number; height: number } }
  | { type: "set_z"; item_id: string; z: number }
  | { type: "update_note"; item_id: string; content: string }
  | { type: "lock_item"; item_id: string; locked: boolean };

function itemMissingDiagnostic(itemId: string) {
  return {
    id: `board-item-missing-${itemId}`,
    code: "CORE/BOARD_ITEM_MISSING",
    severity: "error",
    source: "core",
    message: "board item does not exist",
    target: "board",
  };
}

function itemLockedDiagnostic(itemId: string) {
  return {
    id: `board-item-locked-${itemId}`,
    code: "CORE/BOARD_ITEM_LOCKED",
    severity: "error",
    source: "core",
    message: "cannot modify locked board item",
    target: "board",
  };
}

function evaluateBoardCommands(
  currentItems: BoardItem[],
  commands: RawBoardCommand[],
): { nextItems: BoardItem[]; changes: unknown[]; diagnostics: unknown[] } {
  const nextItems: BoardItem[] = JSON.parse(JSON.stringify(currentItems));
  const diagnostics: unknown[] = [];
  const changes: unknown[] = [];

  for (const cmd of commands) {
    if (!cmd || typeof cmd !== "object") continue;
    switch (cmd.type) {
      case "add_item": {
        const { item_id, kind, position, size } = cmd;
        if (nextItems.some((i) => i.id === item_id)) {
          diagnostics.push({
            id: `board-item-duplicate-${item_id}`,
            code: "CORE/BOARD_ITEM_DUPLICATE",
            severity: "error",
            source: "core",
            message: "board item id already exists",
            target: "board",
          });
          break;
        }
        nextItems.push({
          id: item_id,
          kind,
          position: { x: position?.x ?? 0, y: position?.y ?? 0 },
          size: { width: size?.width ?? 0, height: size?.height ?? 0 },
          z: 0,
          locked: false,
        });
        changes.push({ type: "item_added", item_id });
        break;
      }
      case "remove_item": {
        const { item_id } = cmd;
        const idx = nextItems.findIndex((i) => i.id === item_id);
        if (idx === -1) {
          diagnostics.push(itemMissingDiagnostic(item_id));
          break;
        }
        nextItems.splice(idx, 1);
        changes.push({ type: "item_removed", item_id });
        break;
      }
      case "move_item": {
        const { item_id, position } = cmd;
        const item = nextItems.find((i) => i.id === item_id);
        if (!item) {
          diagnostics.push(itemMissingDiagnostic(item_id));
          break;
        }
        if (item.locked) {
          diagnostics.push(itemLockedDiagnostic(item_id));
          break;
        }
        if (item.position.x === position.x && item.position.y === position.y) {
          break;
        }
        item.position = { x: position.x, y: position.y };
        changes.push({ type: "item_moved", item_id, position });
        break;
      }
      case "resize_item": {
        const { item_id, size } = cmd;
        const item = nextItems.find((i) => i.id === item_id);
        if (!item) {
          diagnostics.push(itemMissingDiagnostic(item_id));
          break;
        }
        if (item.locked) {
          diagnostics.push(itemLockedDiagnostic(item_id));
          break;
        }
        if (item.size.width === size.width && item.size.height === size.height) {
          break;
        }
        item.size = { width: size.width, height: size.height };
        changes.push({ type: "item_resized", item_id, size });
        break;
      }
      case "set_z": {
        const { item_id, z } = cmd;
        const item = nextItems.find((i) => i.id === item_id);
        if (!item) {
          diagnostics.push(itemMissingDiagnostic(item_id));
          break;
        }
        if (item.z === z) {
          break;
        }
        item.z = z;
        changes.push({ type: "item_z_changed", item_id, z });
        break;
      }
      case "update_note": {
        const { item_id, content } = cmd;
        const item = nextItems.find((i) => i.id === item_id);
        if (!item) {
          diagnostics.push(itemMissingDiagnostic(item_id));
          break;
        }
        if (item.locked) {
          diagnostics.push(itemLockedDiagnostic(item_id));
          break;
        }
        if (item.kind.type !== "note") {
          diagnostics.push({
            id: `board-item-not-note-${item_id}`,
            code: "CORE/BOARD_ITEM_NOT_NOTE",
            severity: "error",
            source: "core",
            message: "board item is not a note",
            target: "board",
          });
          break;
        }
        if ((item.kind as { type: "note"; content: string }).content === content) {
          break;
        }
        (item.kind as { type: "note"; content: string }).content = content;
        changes.push({ type: "note_updated", item_id, content });
        break;
      }
      case "lock_item": {
        const { item_id, locked } = cmd;
        const item = nextItems.find((i) => i.id === item_id);
        if (!item) {
          diagnostics.push(itemMissingDiagnostic(item_id));
          break;
        }
        if (item.locked === locked) {
          break;
        }
        item.locked = locked;
        changes.push({ type: "item_locked", item_id, locked });
        break;
      }
    }
  }

  return { nextItems, changes, diagnostics };
}

export async function mockPreviewBoardCommands(input: {
  projectId: string;
  commandBatch: unknown;
}): Promise<BoardCommandResultDto> {
  const board = await mockGetBoardSnapshot({ projectId: input.projectId });
  const batch = input.commandBatch as
    | {
        base_version?: number;
        commands?: RawBoardCommand[];
      }
    | null
    | undefined;

  if (batch?.base_version !== undefined && batch.base_version !== board.version) {
    return {
      status: "rejected",
      version: board.version,
      changes: [],
      diagnostics: [
        {
          id: `board-version-conflict-${board.id}-${batch.base_version}`,
          code: "CORE/BOARD_VERSION_CONFLICT",
          severity: "error",
          source: "core",
          message: `board version conflict: expected ${board.version}, got ${batch.base_version}`,
          target: "board",
        },
      ],
      historyEntryId: null,
    };
  }

  const commands = batch?.commands ?? [];
  if (commands.length === 0) {
    return {
      status: "no_op",
      version: board.version,
      changes: [],
      diagnostics: [],
      historyEntryId: null,
    };
  }

  const { changes, diagnostics } = evaluateBoardCommands(board.items, commands);
  if (diagnostics.length > 0) {
    return {
      status: "rejected",
      version: board.version,
      changes: [],
      diagnostics,
      historyEntryId: null,
    };
  }

  if (changes.length === 0) {
    return {
      status: "no_op",
      version: board.version,
      changes: [],
      diagnostics: [],
      historyEntryId: null,
    };
  }

  return {
    status: "applied",
    version: board.version + 1,
    changes,
    diagnostics: [],
    historyEntryId: null,
  };
}

export async function mockApplyBoardCommands(input: {
  projectId: string;
  commandBatch: unknown;
}): Promise<BoardCommandResultDto> {
  const board = await mockGetBoardSnapshot({ projectId: input.projectId });
  const history =
    mockBoardHistory[input.projectId] ??
    (mockBoardHistory[input.projectId] = { past: [], future: [] });

  const batch = input.commandBatch as
    | {
        id?: string;
        base_version?: number;
        commands?: RawBoardCommand[];
      }
    | null
    | undefined;

  if (batch?.base_version !== undefined && batch.base_version !== board.version) {
    return {
      status: "rejected",
      version: board.version,
      changes: [],
      diagnostics: [
        {
          id: `board-version-conflict-${board.id}-${batch.base_version}`,
          code: "CORE/BOARD_VERSION_CONFLICT",
          severity: "error",
          source: "core",
          message: `board version conflict: expected ${board.version}, got ${batch.base_version}`,
          target: "board",
        },
      ],
      historyEntryId: null,
    };
  }

  const commands = batch?.commands ?? [];
  if (commands.length === 0) {
    return {
      status: "no_op",
      version: board.version,
      changes: [],
      diagnostics: [],
      historyEntryId: null,
    };
  }

  const { nextItems, changes, diagnostics } = evaluateBoardCommands(board.items, commands);
  if (diagnostics.length > 0) {
    return {
      status: "rejected",
      version: board.version,
      changes: [],
      diagnostics,
      historyEntryId: null,
    };
  }

  if (changes.length === 0) {
    return {
      status: "no_op",
      version: board.version,
      changes: [],
      diagnostics: [],
      historyEntryId: null,
    };
  }

  const entryId = `board-history:${batch?.id ?? rand("batch")}`;
  // Mirrors the host: one target per changed item, in change order.
  const targets = [
    ...new Set(
      changes
        .map((change) => (change as { item_id?: string }).item_id)
        .filter((id): id is string => typeof id === "string"),
    ),
  ];
  history.past.push({
    items: JSON.parse(JSON.stringify(board.items)),
    entryId,
    targets,
  });
  history.future = [];
  board.items = nextItems;
  board.version += 1;

  emitEvent({
    type: "document",
    kind: "board.changed",
    projectId: input.projectId,
    documentId: board.id,
    version: board.version,
  });

  const { summaryKey, summary } = mockApplySummary(commands);
  appendMockActivity(input.projectId, {
    actor: { kind: "Human", id: null, label: null },
    summary,
    summaryKey,
    summaryParams: { count: commands.length },
    targets,
    kind: "applied",
    historyEntryId: entryId,
  });

  return {
    status: "applied",
    version: board.version,
    changes,
    diagnostics: [],
    historyEntryId: entryId,
  };
}

export async function mockUndoBoard(input: {
  projectId: string;
  expectedEntryId?: string | null;
}): Promise<BoardCommandResultDto | null> {
  const board = await mockGetBoardSnapshot({ projectId: input.projectId });
  const history = mockBoardHistory[input.projectId];
  const expected = input.expectedEntryId;
  if (expected != null && mockBoardEntryId(history, "past") !== expected) {
    rejectBoardHistoryMismatch(input.projectId, "undo", expected, history);
  }
  if (!history || history.past.length === 0) {
    return null;
  }
  const entry = history.past.pop()!;
  history.future.push({
    items: JSON.parse(JSON.stringify(board.items)),
    entryId: entry.entryId,
    targets: entry.targets,
  });
  board.items = entry.items;
  board.version += 1;

  emitEvent({
    type: "document",
    kind: "board.changed",
    projectId: input.projectId,
    documentId: board.id,
    version: board.version,
  });

  appendMockActivity(input.projectId, {
    actor: { kind: "Human", id: null, label: null },
    summary: "Undid the last board change",
    summaryKey: "activity.board.undo",
    summaryParams: {},
    targets: entry.targets,
    kind: "undone",
    historyEntryId: entry.entryId,
  });

  return {
    status: "applied",
    version: board.version,
    changes: [],
    diagnostics: [],
    historyEntryId: entry.entryId,
  };
}

export async function mockRedoBoard(input: {
  projectId: string;
  expectedEntryId?: string | null;
}): Promise<BoardCommandResultDto | null> {
  const board = await mockGetBoardSnapshot({ projectId: input.projectId });
  const history = mockBoardHistory[input.projectId];
  const expected = input.expectedEntryId;
  if (expected != null && mockBoardEntryId(history, "future") !== expected) {
    rejectBoardHistoryMismatch(input.projectId, "redo", expected, history);
  }
  if (!history || history.future.length === 0) {
    return null;
  }
  const entry = history.future.pop()!;
  history.past.push({
    items: JSON.parse(JSON.stringify(board.items)),
    entryId: entry.entryId,
    targets: entry.targets,
  });
  board.items = entry.items;
  board.version += 1;

  emitEvent({
    type: "document",
    kind: "board.changed",
    projectId: input.projectId,
    documentId: board.id,
    version: board.version,
  });

  appendMockActivity(input.projectId, {
    actor: { kind: "Human", id: null, label: null },
    summary: "Redid the last board change",
    summaryKey: "activity.board.redo",
    summaryParams: {},
    targets: entry.targets,
    kind: "redone",
    historyEntryId: entry.entryId,
  });

  return {
    status: "applied",
    version: board.version,
    changes: [],
    diagnostics: [],
    historyEntryId: entry.entryId,
  };
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
): Promise<ModelDownloadOutput> {
  const total = 6_940_000_000;
  const chunk = 500_000_000;
  let bytes = 0;
  const repoId = args.repoId;

  const emit = (stage: string, message: string | null = null) => {
    const payload: DownloadProgress = {
      downloadId: args.downloadId,
      bytesDownloaded: bytes,
      totalBytes: total,
      message,
    };
    emitEvent({ type: "mod", kind: `aigc.download.${stage}`, payload });
  };

  emit("started");
  while (bytes < total) {
    await delay(80);
    bytes = Math.min(total, bytes + chunk);
    emit("progress", "Downloading model files");
  }
  emit("completed");

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

/* ───── Operation Orchestration Mocks (W-a / W-b) ───── */

export type MockCollectionDoc = {
  id: string;
  projectId: string;
  name: string;
  version: number;
  slots: Array<{ id: string; asset_id: string }>;
};

type MockAsset = {
  projectId: string;
  mediaType: string;
  content: string;
};

/** Project that owns the dev fixture library, mirroring the seeded Default project. */
const MOCK_FIXTURE_PROJECT_ID = "default";

/** Dev fixture collection so the preview library has a collection row. */
const mockCollections: Record<string, MockCollectionDoc> = {
  col_demo_drafts: {
    id: "col_demo_drafts",
    projectId: MOCK_FIXTURE_PROJECT_ID,
    name: "Blog drafts",
    version: 4,
    slots: [
      { id: "slot-1", asset_id: "ast_demo_style_guide" },
      { id: "slot-2", asset_id: "ast_demo_blog_v2" },
      { id: "slot-3", asset_id: "ast_demo_cover_brief" },
    ],
  },
};
const mockCollectionHistory: Record<
  string,
  { past: MockCollectionDoc[]; future: MockCollectionDoc[] }
> = {};
/** Dev fixture files so the preview library has file rows. */
const mockAssets: Record<string, MockAsset> = {
  ast_demo_style_guide: {
    projectId: MOCK_FIXTURE_PROJECT_ID,
    mediaType: "text/markdown",
    content:
      "# Style guide\n\nVoice, headings and image ratios for the workspace.\n\n" +
      "Use sentence case for headings and plain verbs for actions.\n".repeat(40),
  },
  ast_demo_blog_v2: {
    projectId: MOCK_FIXTURE_PROJECT_ID,
    mediaType: "text/markdown",
    content:
      "# Blog v2\n\nOutline and opening paragraph.\n\n" +
      "The draft keeps one idea per section and cites its source.\n".repeat(28),
  },
  ast_demo_cover_brief: {
    projectId: MOCK_FIXTURE_PROJECT_ID,
    mediaType: "text/plain",
    content:
      "Cover brief: warm grey background, one image, no headline.\n" +
      "Keep the lower third clear for the series label.\n".repeat(55),
  },
};

const MOCK_DYNAMIC_WORKFLOWS = [
  {
    id: "dwf_demo_cleanup",
    projectId: MOCK_FIXTURE_PROJECT_ID,
    name: "Draft cleanup",
    version: 3,
    step_count: 3,
  },
  {
    id: "dwf_demo_archive",
    projectId: MOCK_FIXTURE_PROJECT_ID,
    name: "Archive sweep",
    version: 1,
    step_count: 2,
  },
];

const MOCK_DEMO_RUN_ID = "run_2";

function mockDemoRun(projectId: string) {
  const now = Date.now();
  return {
    id: MOCK_DEMO_RUN_ID,
    project_id: projectId,
    dynamic_workflow_id: "dwf_demo_cleanup",
    target: { collection_id: "col_demo_drafts" },
    params: {},
    state: "completed",
    version: 1,
    items: [
      { item_id: "slot-1", slot_id: "slot-1", state: "succeeded" },
      { item_id: "slot-2", slot_id: "slot-2", state: "succeeded" },
      { item_id: "slot-3", slot_id: "slot-3", state: "succeeded" },
    ],
    collection_id: "col_demo_drafts",
    created_at: new Date(now - 6 * 60_000).toISOString(),
    finished_at: new Date(now - 4 * 60_000).toISOString(),
  };
}

export async function mockReadAssetPreview(
  url: string,
  assetId: string,
  maxLines: number,
): Promise<{ url: string; mediaType: string; text: string | null }> {
  const asset = mockAssets[assetId];
  if (!asset) throw new Error(`mock asset ${assetId} not found`);
  const textual = asset.mediaType.startsWith("text/") || asset.mediaType === "application/json";
  return {
    url,
    mediaType: asset.mediaType,
    text: textual ? asset.content.split("\n").slice(0, maxLines).join("\n") : null,
  };
}

export async function mockOpInvoke(input: {
  projectId: string;
  name: string;
  input: unknown;
}): Promise<InvokeResult> {
  const { projectId, name, input: opInput } = input;
  const args = (opInput as Record<string, unknown>) ?? {};

  if (name === "asset.import") {
    const paths = (args.paths as string[]) ?? [];
    const assetIds: string[] = [];
    for (const p of paths) {
      const assetId = rand("ast");
      assetIds.push(assetId);
      const filename = p.split(/[/\\]/).pop() ?? "file.txt";
      mockAssets[assetId] = {
        projectId,
        mediaType: filename.endsWith(".png")
          ? "image/png"
          : filename.endsWith(".json")
            ? "application/json"
            : "text/plain",
        content: `Preview of ${filename}\nImported from ${p}\nLine 3 sample text content`,
      };
    }
    return {
      step: {
        id: rand("step"),
        op: "asset.import",
        outcome: "succeeded",
      },
      output: { assets: assetIds },
    };
  }

  if (name === "collection.create") {
    const colName = (args.name as string) || "Untitled Collection";
    const assets = (args.assets as string[]) || [];
    const collectionId = rand("col");
    const boardItemId = rand("item");

    const slots = assets.map((asset, idx) => ({
      id: `slot-${idx + 1}`,
      asset_id: asset,
    }));

    const doc: MockCollectionDoc = {
      id: collectionId,
      projectId,
      name: colName,
      version: 1,
      slots,
    };
    mockCollections[collectionId] = doc;
    mockCollectionHistory[collectionId] = { past: [], future: [] };

    const board = await mockGetBoardSnapshot({ projectId });
    board.version += 1;
    board.items.push({
      id: boardItemId,
      kind: { type: "collection", collection_id: collectionId },
      position: { x: 0, y: 0 },
      size: { width: 300, height: 200 },
      z: board.items.length + 1,
      locked: false,
    });

    emitEvent({
      type: "document",
      kind: "collection.changed",
      projectId,
      documentId: collectionId,
      version: 1,
    });
    emitEvent({
      type: "document",
      kind: "board.changed",
      projectId,
      documentId: board.id,
      version: board.version,
    });

    return {
      step: { id: rand("step"), op: "collection.create", outcome: "succeeded" },
      output: {
        collection_id: collectionId,
        board_item_id: boardItemId,
      },
    };
  }

  if (name === "collection.get") {
    const colId = (args.collection_id as string) || "";
    const doc = mockCollections[colId];
    if (!doc) {
      throw new Error(`COLLECTION/CALLER_UNKNOWN_COLLECTION: collection ${colId} not found`);
    }
    return {
      step: { id: rand("step"), op: "collection.get", outcome: "succeeded" },
      output: {
        id: doc.id,
        name: doc.name,
        version: doc.version,
        slots: doc.slots,
      },
    };
  }

  if (name === "collection.set_slot") {
    const colId = (args.collection_id as string) || "";
    const slotId = (args.slot_id as string) || "";
    const asset = (args.asset as string) || "";
    const doc = mockCollections[colId];
    if (!doc) {
      throw new Error(`COLLECTION/CALLER_UNKNOWN_COLLECTION: collection ${colId} not found`);
    }
    const history = mockCollectionHistory[colId] || { past: [], future: [] };
    history.past.push(JSON.parse(JSON.stringify(doc)));
    history.future = [];

    const slot = doc.slots.find((s) => s.id === slotId);
    if (!slot) {
      throw new Error(`COLLECTION/CALLER_UNKNOWN_SLOT: slot ${slotId} not found`);
    }
    slot.asset_id = asset;
    doc.version += 1;

    emitEvent({
      type: "document",
      kind: "collection.changed",
      projectId,
      documentId: colId,
      version: doc.version,
    });

    return {
      step: { id: rand("step"), op: "collection.set_slot", outcome: "succeeded" },
      output: { version: doc.version },
    };
  }

  if (name === "collection.undo") {
    const colId = (args.collection_id as string) || "";
    const doc = mockCollections[colId];
    if (!doc) {
      throw new Error(`COLLECTION/CALLER_UNKNOWN_COLLECTION: collection ${colId} not found`);
    }
    const history = mockCollectionHistory[colId] || { past: [], future: [] };
    const prev = history.past.pop();
    if (prev) {
      history.future.push(JSON.parse(JSON.stringify(doc)));
      doc.name = prev.name;
      doc.slots = prev.slots;
      doc.version += 1;

      emitEvent({
        type: "document",
        kind: "collection.changed",
        projectId,
        documentId: colId,
        version: doc.version,
      });
      return {
        step: { id: rand("step"), op: "collection.undo", outcome: "succeeded" },
        output: { applied: true, version: doc.version },
      };
    }
    return {
      step: { id: rand("step"), op: "collection.undo", outcome: "succeeded" },
      output: { applied: false, version: doc.version },
    };
  }

  if (name === "collection.redo") {
    const colId = (args.collection_id as string) || "";
    const doc = mockCollections[colId];
    if (!doc) {
      throw new Error(`COLLECTION/CALLER_UNKNOWN_COLLECTION: collection ${colId} not found`);
    }
    const history = mockCollectionHistory[colId] || { past: [], future: [] };
    const next = history.future.pop();
    if (next) {
      history.past.push(JSON.parse(JSON.stringify(doc)));
      doc.name = next.name;
      doc.slots = next.slots;
      doc.version += 1;

      emitEvent({
        type: "document",
        kind: "collection.changed",
        projectId,
        documentId: colId,
        version: doc.version,
      });
      return {
        step: { id: rand("step"), op: "collection.redo", outcome: "succeeded" },
        output: { applied: true, version: doc.version },
      };
    }
    return {
      step: { id: rand("step"), op: "collection.redo", outcome: "succeeded" },
      output: { applied: false, version: doc.version },
    };
  }

  if (name === "collection.list") {
    const collections = Object.values(mockCollections)
      .filter((doc) => doc.projectId === projectId)
      .map((doc) => ({
        id: doc.id,
        name: doc.name,
        version: doc.version,
        slot_count: doc.slots.length,
      }));
    return {
      step: { id: rand("step"), op: name, outcome: "succeeded" },
      output: { collections },
    };
  }

  if (name === "dynamic_workflow.list") {
    const workflows = MOCK_DYNAMIC_WORKFLOWS.filter(
      (workflow) => workflow.projectId === projectId,
    ).map((workflow) => ({
      id: workflow.id,
      name: workflow.name,
      version: workflow.version,
      step_count: workflow.step_count,
    }));
    return {
      step: { id: rand("step"), op: name, outcome: "succeeded" },
      output: { workflows },
    };
  }

  if (name === "asset.list") {
    const assets = Object.entries(mockAssets)
      .reverse()
      .filter(([, asset]) => asset.projectId === projectId)
      .map(([id, asset]) => ({
        id,
        media_type: asset.mediaType,
        size: asset.content.length,
        origin: "step-mock",
      }));
    return {
      step: { id: rand("step"), op: name, outcome: "succeeded" },
      output: { assets },
    };
  }

  if (name === "activity.list") {
    const limit = typeof args.limit === "number" ? Math.max(1, Math.floor(args.limit)) : 50;
    const before = typeof args.before === "string" ? args.before : null;
    const all = mockActivityFor(projectId);
    const start = before ? all.findIndex((record) => record.id === before) + 1 : 0;
    return {
      step: { id: rand("step"), op: name, outcome: "succeeded" },
      output: {
        records: all.slice(start, start + limit),
        has_more: start + limit < all.length,
      },
    };
  }

  if (name === "dynamic_workflow.list_runs") {
    const collectionId = args.collection_id as string | undefined;
    const limit = typeof args.limit === "number" ? Math.max(1, args.limit) : 50;
    // The demo run belongs to the fixture project's collection; other projects
    // and other collections have no runs, matching the old empty behavior.
    const hasDemoRun =
      projectId === MOCK_FIXTURE_PROJECT_ID &&
      (!collectionId || collectionId === "col_demo_drafts");
    if (!hasDemoRun) {
      return {
        step: { id: rand("step"), op: name, outcome: "succeeded" },
        output: { runs: [] },
      };
    }
    return {
      step: { id: rand("step"), op: name, outcome: "succeeded" },
      output: { runs: [mockDemoRun(projectId)].slice(0, limit) },
    };
  }

  return {
    step: { id: rand("step"), op: name, outcome: "succeeded" },
    output: {},
  };
}
