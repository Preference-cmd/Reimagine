import { z } from "zod";

/* ───── Socket kinds (mirror of ui/src/design/tokens.ts) ───── */

export const SocketKindSchema = z.enum([
  "model",
  "conditioning",
  "latent",
  "image",
]);
export type SocketKind = z.infer<typeof SocketKindSchema>;

/* ───── Socket spec (port on a node) ───── */

export const SocketSpecSchema = z.object({
  id: z.string(),
  kind: SocketKindSchema,
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
});
export type ParamSpec = z.infer<typeof ParamSpecSchema>;

/* ───── Node definition (registry payload from Rust) ───── */

export const NodeCategorySchema = z.enum([
  "loaders",
  "conditioning",
  "latent",
  "sampling",
  "vae",
  "output",
]);
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
  diagnostics: z.array(z.object({
    id: z.string(),
    code: z.string(),
    severity: z.string(),
    source: z.string(),
    message: z.string(),
    target: z.string(),
  })),
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
    diagnostics: z.array(z.object({
      id: z.string(),
      code: z.string(),
      severity: z.string(),
      source: z.string(),
      message: z.string(),
      target: z.string(),
    })),
  }),
  z.object({
    outcome: z.literal("blocked"),
    workflowId: z.string(),
    diagnostics: z.array(z.object({
      id: z.string(),
      code: z.string(),
      severity: z.string(),
      source: z.string(),
      message: z.string(),
      target: z.string(),
    })),
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