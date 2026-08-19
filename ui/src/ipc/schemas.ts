import { z } from "zod";

/* ───── Socket kinds (mirror of ui/src/design/tokens.ts) ───── */

export const SocketKindSchema = z.enum(["model", "conditioning", "latent", "image"]);
export type SocketKind = z.infer<typeof SocketKindSchema>;

/* ───── Socket spec (port on a node) ───── */

export const SocketSpecSchema = z.object({
  id: z.string(),
  // Open string: the backend catalog emits kinds beyond the four socket
  // colors (clip, vae, model_ref, string, artifact, …). The canvas maps
  // these to the closest visual token at render time.
  kind: z.string(),
  label: z.string(),
});
export type SocketSpec = z.infer<typeof SocketSpecSchema>;

/* ───── Parameter spec ───── */

export const ParamKindSchema = z.enum([
  "int",
  "float",
  "string",
  "select",
  "bool",
  "text",
  "image",
]);
export type ParamKind = z.infer<typeof ParamKindSchema>;

export const ParamSpecSchema = z.object({
  id: z.string(),
  label: z.string(),
  kind: ParamKindSchema,
  default: z.unknown().optional(),
  options: z.array(z.string()).optional(),
  min: z.number().optional(),
  max: z.number().optional(),
  step: z.number().optional(),
});
export type ParamSpec = z.infer<typeof ParamSpecSchema>;

/* ───── Node definition (registry payload from Rust) ───── */

export const NodeCategorySchema = z.string();
export type NodeCategory = z.infer<typeof NodeCategorySchema>;

export const NodeDefSchema = z.object({
  type: z.string(),
  displayName: z.string(),
  category: NodeCategorySchema,
  inputs: z.array(SocketSpecSchema),
  outputs: z.array(SocketSpecSchema),
  parameters: z.array(ParamSpecSchema),
});
export type NodeDef = z.infer<typeof NodeDefSchema>;

/* ───── Workflow payload (sent to runWorkflow) ───── */

export const WorkflowNodeSchema = z.object({
  id: z.string(),
  type: z.string(),
  position: z.object({ x: z.number(), y: z.number() }),
  data: z.record(z.string(), z.unknown()),
});
export type WorkflowNode = z.infer<typeof WorkflowNodeSchema>;

export const WorkflowEdgeSchema = z.object({
  id: z.string(),
  source: z.string(),
  sourceHandle: z.string().nullable(),
  target: z.string(),
  targetHandle: z.string().nullable(),
});
export type WorkflowEdge = z.infer<typeof WorkflowEdgeSchema>;

export const WorkflowSchema = z.object({
  nodes: z.array(WorkflowNodeSchema),
  edges: z.array(WorkflowEdgeSchema),
});
export type Workflow = z.infer<typeof WorkflowSchema>;

/* ───── Saved workflow summaries (from `list_workflows`) ───── */

export const WorkflowFileSummarySchema = z.object({
  id: z.string(),
  modified_millis: z.number(),
});
export type WorkflowFileSummary = z.infer<typeof WorkflowFileSummarySchema>;


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

export const BoardItemSchema = z.object({
  id: z.string(),
  kind: z.record(z.string(), z.unknown()),
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
  boardVersion: z.number(),
  changes: z.array(z.unknown()),
  diagnostics: z.array(z.unknown()),
  historyEntryId: z.string().nullable().optional(),
});
export type BoardCommandResult = z.infer<typeof BoardCommandResultSchema>;

export const DocumentChangedEventSchema = z.object({
  kind: z.enum(["board.changed", "workflow.changed"]),
  projectId: z.string(),
  documentId: z.string(),
  version: z.number(),
});
export type DocumentChangedEvent = z.infer<typeof DocumentChangedEventSchema>;

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

/* ───── Run events from Rust IPC ───── */

export const RunEventPayloadSchema = z.object({
  id: z.string(),
  runId: z.string(),
  kind: z.string(),
  nodeId: z.string().nullable(),
  artifactId: z.string().nullable(),
  createdAt: z.string(),
});
export type RunEventPayload = z.infer<typeof RunEventPayloadSchema>;

export const RunSnapshotDtoSchema = z.object({
  runId: z.string(),
  workflowId: z.string(),
  state: z.string(),
  nodeStates: z.record(z.string(), z.string()),
  diagnostics: z.array(
    z.object({
      id: z.string(),
      code: z.string(),
      severity: z.string(),
      source: z.string(),
      message: z.string(),
      target: z.string(),
    }),
  ),
  artifacts: z.array(z.any()),
  startedAt: z.string(),
  updatedAt: z.string(),
});
export type RunSnapshotDto = z.infer<typeof RunSnapshotDtoSchema>;

export const RunWorkflowResponseSchema = z.discriminatedUnion("outcome", [
  z.object({
    outcome: z.literal("started"),
    runId: z.string(),
    workflowId: z.string(),
    workflowVersion: z.string(),
    initialSnapshot: RunSnapshotDtoSchema,
    diagnostics: z.array(
      z.object({
        id: z.string(),
        code: z.string(),
        severity: z.string(),
        source: z.string(),
        message: z.string(),
        target: z.string(),
      }),
    ),
  }),
  z.object({
    outcome: z.literal("blocked"),
    workflowId: z.string(),
    diagnostics: z.array(
      z.object({
        id: z.string(),
        code: z.string(),
        severity: z.string(),
        source: z.string(),
        message: z.string(),
        target: z.string(),
      }),
    ),
  }),
]);
export type RunWorkflowResponse = z.infer<typeof RunWorkflowResponseSchema>;

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

export const DownloadEventPayloadSchema = z.object({
  id: z.string(),
  status: z.string(),
  repoId: z.string(),
  revision: z.string(),
  bytesDownloaded: z.number(),
  totalBytes: z.number().nullable().optional(),
  message: z.string().nullable().optional(),
  modelName: z.string().optional(),
  detectedFormat: z.string().optional(),
  estimatedSize: z.number().optional(),
});
export type DownloadEventPayload = z.infer<typeof DownloadEventPayloadSchema>;

export const DownloadHuggingfaceModelArgsSchema = z.object({
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

/* ───── Worker switching (BE-32) ───── */

/** Active worker after a switch (mirrors `WorkerSwitchResultDto`). */
export const WorkerSwitchResultSchema = z.object({
  instance: z.string(),
  incarnationId: z.string(),
});
export type WorkerSwitchResult = z.infer<typeof WorkerSwitchResultSchema>;

/** Installed worker usable as a switch target (mirrors `WorkerInstallationDto`). */
export const WorkerSwitchTargetSchema = z.object({
  installationId: z.string(),
  version: z.string(),
  backendInstanceId: z.string(),
  backendKind: z.string(),
  target: z.string(),
  installedAt: z.string(),
  installPath: z.string(),
  manifestDigest: z.string(),
});
export type WorkerSwitchTarget = z.infer<typeof WorkerSwitchTargetSchema>;

export const WorkerSwitchArgsSchema = z.object({
  /** Backend instance id of the installed worker, e.g. `burn:wgpu:default`. */
  target: z.string(),
  /** Drain/cancel deadline in seconds (defaults to 30 on the backend). */
  deadlineSecs: z.number().positive().optional(),
});
export type WorkerSwitchArgs = z.infer<typeof WorkerSwitchArgsSchema>;

/* ───── Backend re-bootstrap (BE-38 / B4-8) ───── */

/** Backend kind accepted by `rebootstrap_backend`. */
export const BackendSelectionSchema = z.enum(["burn", "candle"]);
export type BackendSelection = z.infer<typeof BackendSelectionSchema>;

export const RebootBackendArgsSchema = z.object({
  selection: BackendSelectionSchema,
});
export type RebootBackendArgs = z.infer<typeof RebootBackendArgsSchema>;

/** Compute profile returned by `rebootstrap_backend` (mirrors
 *  `ComputeProfileDto`; wire shape is snake_case and intentionally loose —
 *  the UI reads it for display only). */
export const ComputeProfileSchema = z.object({
  backend_profiles: z.array(z.any()),
  diagnostics: z.array(z.any()),
  /** Topology pool workers (T13): discovered QUIC/gRPC endpoints that
   *  are registered but not necessarily connected. Optional — absent
   *  when no topology manager is configured. */
  topology_workers: z
    .array(
      z.object({
        id: z.string(),
        transport: z.string(),
        address: z.string(),
        trusted: z.boolean(),
        device_label: z.string(),
        capabilities: z.array(z.string()),
        state: z.string(),
      }),
    )
    .optional()
    .default([]),
});
export type ComputeProfile = z.infer<typeof ComputeProfileSchema>;
