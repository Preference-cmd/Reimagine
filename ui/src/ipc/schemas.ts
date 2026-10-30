import { z } from "zod";

/* ───── Project / Board surface (AR-39) ─────────────────────── */

export const ProjectSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string(),
  createdAt: z.string(),
  updatedAt: z.string(),
});
export type Project = z.infer<typeof ProjectSchema>;

export const ProjectMetadataInputSchema = z.object({
  name: z.string(),
  description: z.string(),
  createdAt: z.string().optional(),
  updatedAt: z.string().optional(),
});
export type ProjectMetadataInput = z.infer<typeof ProjectMetadataInputSchema>;

export const BoardItemKindWorkflowRefSchema = z.object({
  type: z.literal("workflow_ref"),
  workflow_id: z.string(),
});
export const BoardItemKindAssetRefSchema = z.object({
  type: z.literal("asset_ref"),
  artifact_id: z.string(),
});
export const BoardItemKindNoteSchema = z.object({
  type: z.literal("note"),
  content: z.string(),
});
export const BoardItemKindRunRefSchema = z.object({
  type: z.literal("run_ref"),
  run_id: z.string(),
});
export const BoardItemKindCollectionSchema = z.object({
  type: z.literal("collection"),
  collection_id: z.string(),
});

export const BoardItemKindSchema = z.discriminatedUnion("type", [
  BoardItemKindWorkflowRefSchema,
  BoardItemKindAssetRefSchema,
  BoardItemKindNoteSchema,
  BoardItemKindRunRefSchema,
  BoardItemKindCollectionSchema,
]);
export type BoardItemKind = z.infer<typeof BoardItemKindSchema>;

export const KNOWN_BOARD_ITEM_KINDS = [
  "workflow_ref",
  "asset_ref",
  "note",
  "run_ref",
  "collection",
] as const;

export const BoardItemUnknownKindSchema = z
  .object({
    type: z
      .string()
      .refine(
        (val) => !KNOWN_BOARD_ITEM_KINDS.includes(val as (typeof KNOWN_BOARD_ITEM_KINDS)[number]),
        {
          message: "Known board item kinds must match their specific schema",
        },
      ),
  })
  .passthrough();

export const BoardItemSchema = z.object({
  id: z.string(),
  kind: z.union([BoardItemKindSchema, BoardItemUnknownKindSchema]),
  position: z.object({ x: z.number(), y: z.number() }),
  size: z.object({ width: z.number(), height: z.number() }),
  z: z.number(),
  locked: z.boolean(),
});
export type BoardItem = z.infer<typeof BoardItemSchema>;

export const BoardSnapshotSchema = z.object({
  id: z.string(),
  projectId: z.string(),
  version: z.number(),
  items: z.array(BoardItemSchema),
});
export type BoardSnapshot = z.infer<typeof BoardSnapshotSchema>;

export const BoardCommandResultSchema = z.object({
  status: z.enum(["applied", "rejected", "no_op"]),
  version: z.number(),
  changes: z.array(z.unknown()),
  diagnostics: z.array(z.unknown()),
  historyEntryId: z.string().nullable().optional(),
});
export type BoardCommandResultDto = z.infer<typeof BoardCommandResultSchema>;

/* ───── Events from `subscribe_events` ───── */

export const DocumentEventSchema = z.object({
  type: z.literal("document"),
  kind: z.string(),
  projectId: z.string(),
  documentId: z.string(),
  version: z.number(),
});
export type DocumentEvent = z.infer<typeof DocumentEventSchema>;

export const RunEventSchema = z.object({
  type: z.literal("run"),
  id: z.string(),
  runId: z.string(),
  workflowId: z.string(),
  workflowVersion: z.number(),
  kind: z.string(),
  nodeId: z.string().nullable(),
  artifact: z.string().nullable(),
  diagnostics: z.array(z.unknown()),
  progress: z.unknown().nullable(),
  createdAt: z.string(),
  correlationId: z.string().nullable(),
});
export type RunEvent = z.infer<typeof RunEventSchema>;

export const ModEventSchema = z.object({
  type: z.literal("mod"),
  kind: z.string(),
  payload: z.unknown(),
});
export type ModEvent = z.infer<typeof ModEventSchema>;

export const EventSchema = z.discriminatedUnion("type", [
  DocumentEventSchema,
  RunEventSchema,
  ModEventSchema,
]);
export type Event = z.infer<typeof EventSchema>;

/** Payload of the `aigc.download.*` mod events. */
export const DownloadProgressSchema = z.object({
  downloadId: z.string(),
  /** Bytes of the files finished so far. */
  bytesDownloaded: z.number(),
  totalBytes: z.number().nullable(),
  message: z.string().nullable(),
});
export type DownloadProgress = z.infer<typeof DownloadProgressSchema>;

/* ───── Misc ───── */

export const RunIdSchema = z.string().regex(/^run_[a-z0-9]+$/);
export type RunId = z.infer<typeof RunIdSchema>;

export const ModelInfoSchema = z.object({
  id: z.string(),
  displayName: z.string(),
  modelSeries: z.string(),
  variant: z.string(),
  roles: z.array(z.string()),
  format: z.string(),
  sourceStatus: z.string(),
  sizeBytes: z.number().nullable(),
});
export type ModelInfo = z.infer<typeof ModelInfoSchema>;

/* ───── Artifact metadata from Rust IPC ───── */

export const ArtifactMetadataSchema = z.object({
  id: z.string(),
  nodeId: z.string(),
  mediaType: z.string(),
  filename: z.string(),
  path: z.string(),
});
export type ArtifactMetadata = z.infer<typeof ArtifactMetadataSchema>;

/* ───── Model catalog & download (mirror app-host dto/model_acquisition.rs) ───── */

export const ModelFiltersSchema = z.object({
  pipelineTag: z.string().nullable().optional(),
  libraryName: z.string().nullable().optional(),
  tags: z.array(z.string()).default([]),
  sort: z.enum(["downloads", "likes", "trending", "lastModified"]).default("downloads"),
  limit: z.number().default(20),
});
export type ModelFilters = z.infer<typeof ModelFiltersSchema>;

export const ModelCatalogEntrySchema = z.object({
  id: z.string(),
  author: z.string().nullable().optional(),
  pipelineTag: z.string().nullable().optional(),
  tags: z.array(z.string()),
  downloads: z.number(),
  likes: z.number(),
  lastModified: z.string().nullable().optional(),
  private: z.boolean(),
});
export type ModelCatalogEntry = z.infer<typeof ModelCatalogEntrySchema>;

export const ModelCardSchema = z.object({
  entry: ModelCatalogEntrySchema,
  detectedFormat: z.string(),
  estimatedDownloadSize: z.number(),
  modelSummary: z.string().nullable().optional(),
  fileCount: z.number(),
  components: z.array(z.string()),
});
export type ModelCard = z.infer<typeof ModelCardSchema>;

export const DownloadHuggingfaceModelArgsSchema = z.object({
  downloadId: z.string(),
  repoId: z.string(),
  revision: z.string().optional(),
  allowPatterns: z.array(z.string()).optional(),
  targetRelativeDir: z.string(),
  overwrite: z.string().optional(),
  autoDetect: z.boolean().optional(),
  fromCatalog: z.boolean().optional(),
});
export type DownloadHuggingfaceModelArgs = z.infer<typeof DownloadHuggingfaceModelArgsSchema>;

export const FileEntrySchema = z.object({
  relativePath: z.string(),
  bytes: z.number(),
  outcome: z.string(),
});
export type FileEntry = z.infer<typeof FileEntrySchema>;

export const ModelDownloadOutputSchema = z.object({
  effective: z.boolean(),
  provider: z.string(),
  repoId: z.string(),
  revision: z.string(),
  targetDir: z.string(),
  files: z.array(FileEntrySchema),
  totalBytes: z.number(),
  finishedAt: z.string(),
  detectedFormat: z.string().optional(),
});
export type ModelDownloadOutput = z.infer<typeof ModelDownloadOutputSchema>;

/* ───── Structured command errors (BE-31) ───── */

/** Structured error payload returned by Tauri commands.
 *
 * `code` is the snake_case `AppHostErrorCode` (e.g. `worker_unavailable`,
 * `model_not_found`, `workflow_invalid`) that callers should branch on.
 * `details` carries machine-readable context (ids, instance names) and is
 * absent on legacy payloads and for errors without context. */
export const CommandErrorSchema = z.object({
  code: z.string(),
  message: z.string(),
  details: z.record(z.string(), z.unknown()).nullable().optional(),
});
export type CommandError = z.infer<typeof CommandErrorSchema>;

/* ───── Operation Orchestration (W-a / W-b) ───── */

export const InvokeResultSchema = z.object({
  step: z.unknown(),
  output: z.unknown(),
});
export type InvokeResult = z.infer<typeof InvokeResultSchema>;

export const CollectionSlotSchema = z.object({
  id: z.string(),
  asset_id: z.string(),
});
export type CollectionSlot = z.infer<typeof CollectionSlotSchema>;

export const CollectionDocumentSchema = z.object({
  id: z.string(),
  name: z.string(),
  version: z.number(),
  slots: z.array(CollectionSlotSchema),
});
export type CollectionDocument = z.infer<typeof CollectionDocumentSchema>;

/* ───── Library summaries (collection.list, dynamic_workflow.list, asset.list) ───── */

export const CollectionSummarySchema = z.object({
  id: z.string(),
  name: z.string(),
  version: z.number(),
  slot_count: z.number(),
});
export type CollectionSummary = z.infer<typeof CollectionSummarySchema>;

export const CollectionSummaryListSchema = z.object({
  collections: z.array(CollectionSummarySchema),
});
export type CollectionSummaryList = z.infer<typeof CollectionSummaryListSchema>;

export const DynamicWorkflowSummarySchema = z.object({
  id: z.string(),
  name: z.string(),
  version: z.number(),
  step_count: z.number(),
});
export type DynamicWorkflowSummary = z.infer<typeof DynamicWorkflowSummarySchema>;

export const DynamicWorkflowSummaryListSchema = z.object({
  workflows: z.array(DynamicWorkflowSummarySchema),
});
export type DynamicWorkflowSummaryList = z.infer<typeof DynamicWorkflowSummaryListSchema>;

export const AssetSummarySchema = z.object({
  id: z.string(),
  media_type: z.string(),
  size: z.number(),
  origin: z.string(),
});
export type AssetSummary = z.infer<typeof AssetSummarySchema>;

export const AssetSummaryListSchema = z.object({
  assets: z.array(AssetSummarySchema),
});
export type AssetSummaryList = z.infer<typeof AssetSummaryListSchema>;

/* ───── Dynamic workflow runs (W-c) ───── */

export const RunStateSchema = z.enum([
  "queued",
  "running",
  "completed",
  "failed",
  "cancelled",
  "interrupted",
  "needs_confirmation",
]);
export type RunState = z.infer<typeof RunStateSchema>;

export const SlotRunStateSchema = z.enum([
  "queued",
  "running",
  "succeeded",
  "failed",
  "stale",
  "cancelled",
  "needs_confirmation",
]);
export type SlotRunState = z.infer<typeof SlotRunStateSchema>;

export const RunItemSchema = z.object({
  item_id: z.string(),
  slot_id: z.string().nullable().optional(),
  state: SlotRunStateSchema,
  diagnostic: z.unknown().nullable().optional(),
});
export type RunItem = z.infer<typeof RunItemSchema>;

export const DynamicWorkflowRunSchema = z.object({
  id: z.string(),
  project_id: z.string(),
  dynamic_workflow_id: z.string(),
  target: z.unknown(),
  params: z.unknown().optional(),
  state: RunStateSchema,
  version: z.number(),
  items: z.array(RunItemSchema),
  collection_id: z.string().nullable().optional(),
  created_at: z.string(),
  finished_at: z.string().nullable().optional(),
});
export type DynamicWorkflowRun = z.infer<typeof DynamicWorkflowRunSchema>;

export const DynamicWorkflowRunListSchema = z.object({
  runs: z.array(DynamicWorkflowRunSchema),
});
export type DynamicWorkflowRunList = z.infer<typeof DynamicWorkflowRunListSchema>;

/* ───── Project activity (activity.list, activity.appended) ───── */

/** Mirrors the core `CommandActor` serde: PascalCase kind, optional id/label. */
export const ActivityActorSchema = z.object({
  kind: z.enum(["Human", "Agent", "Importer", "System"]),
  id: z.string().nullable(),
  label: z.string().nullable(),
});
export type ActivityActor = z.infer<typeof ActivityActorSchema>;

export const ActivityKindSchema = z.enum([
  "applied",
  "undone",
  "redone",
  "run_started",
  "run_finished",
  "run_cancelled",
]);
export type ActivityKind = z.infer<typeof ActivityKindSchema>;

/**
 * One append-only activity record. `summary` is the host's English fallback;
 * `summaryKey` + `summaryParams` let clients render their own language.
 */
export const ActivityRecordSchema = z.object({
  id: z.string(),
  at: z.string(),
  actor: ActivityActorSchema,
  summary: z.string(),
  summaryKey: z.string(),
  summaryParams: z.unknown(),
  targets: z.array(z.string()),
  kind: ActivityKindSchema,
  historyEntryId: z.string().nullable(),
});
export type ActivityRecord = z.infer<typeof ActivityRecordSchema>;

/**
 * One `activity.list` page, newest first. The host's operation output spells
 * the paging flag `has_more` (snake_case, unlike the camelCase record body);
 * the camelCase spelling is accepted too and the parse always exposes
 * `hasMore`.
 */
export const ActivityListResultSchema = z
  .object({
    records: z.array(ActivityRecordSchema),
    has_more: z.boolean().optional(),
    hasMore: z.boolean().optional(),
  })
  .transform((value) => ({
    records: value.records,
    hasMore: value.has_more ?? value.hasMore ?? false,
  }));
export type ActivityListResult = z.infer<typeof ActivityListResultSchema>;

/** Payload of `Event::Mod { kind: "activity.appended" }`. */
export const ActivityAppendedPayloadSchema = z.object({
  projectId: z.string(),
  record: ActivityRecordSchema,
});
export type ActivityAppendedPayload = z.infer<typeof ActivityAppendedPayloadSchema>;
