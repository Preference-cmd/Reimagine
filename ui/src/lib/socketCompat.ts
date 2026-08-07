import type { Node } from "@xyflow/react";
import type { NodeDef } from "@/ipc/schemas";

/**
 * Socket kind compatibility (F2-3).
 *
 * Mirrors the backend's `SlotKind::slot_type()` semantics from
 * `crates/core/src/model/slots.rs` — every socket kind classifies into one
 * of five `SlotType`s (Primitive, Tensor, ModelHandle, Conditioning,
 * Artifact) and connections are allowed only between sockets of the same
 * slot type. Kinds the frontend does not recognize are treated as
 * compatible (forward compatibility with new backend kinds).
 */

export type SlotType = "Primitive" | "Tensor" | "ModelHandle" | "Conditioning" | "Artifact";

const PRIMITIVE_KINDS = new Set([
  "string",
  "text",
  "integer",
  "float",
  "bool",
  "seed",
  "select",
  "path",
  "null",
]);
const TENSOR_KINDS = new Set(["latent"]);
const MODEL_HANDLE_KINDS = new Set(["model_ref", "model", "clip", "vae"]);
const CONDITIONING_KINDS = new Set(["conditioning"]);
const ARTIFACT_KINDS = new Set(["image", "artifact"]);

/** Classify a socket kind into its slot type (null when unknown). */
export function socketSlotType(kind: string | null | undefined): SlotType | null {
  if (!kind) return null;
  if (PRIMITIVE_KINDS.has(kind)) return "Primitive";
  if (TENSOR_KINDS.has(kind)) return "Tensor";
  if (MODEL_HANDLE_KINDS.has(kind)) return "ModelHandle";
  if (CONDITIONING_KINDS.has(kind)) return "Conditioning";
  if (ARTIFACT_KINDS.has(kind)) return "Artifact";
  return null;
}

/** Unknown kinds count as compatible so new backend kinds never brick the canvas. */
export function isConnectionAllowed(
  sourceKind: string | null | undefined,
  targetKind: string | null | undefined,
): boolean {
  const sourceType = socketSlotType(sourceKind);
  const targetType = socketSlotType(targetKind);
  if (sourceType == null || targetType == null) return true;
  return sourceType === targetType;
}

/** Human-readable reason when a connection is rejected (null when allowed). */
export function connectionRejectionReason(
  sourceKind: string | null | undefined,
  targetKind: string | null | undefined,
): string | null {
  if (isConnectionAllowed(sourceKind, targetKind)) return null;
  const label = (kind: string | null | undefined) => kind || "?";
  return `Cannot connect ${label(sourceKind)} → ${label(targetKind)}: incompatible socket kinds`;
}

/**
 * Resolve the socket kind for a handle on a node: embedded legacy data
 * (`data.inputs` / `data.outputs`) wins, catalog defs fill the gap for
 * schema-driven nodes. Returns null when the handle is unknown.
 */
export function resolveSocketKind(
  node: Node | undefined,
  def: NodeDef | undefined,
  handleId: string | null | undefined,
  side: "input" | "output",
): string | null {
  if (!node) return null;
  const data = node.data as
    | { inputs?: { id: string; kind?: string }[]; outputs?: { id: string; kind?: string }[] }
    | undefined;
  const pool = side === "input" ? data?.inputs : data?.outputs;
  const embedded = pool?.find((socket) => socket.id === handleId);
  if (embedded?.kind) return embedded.kind;
  const defPool = side === "input" ? def?.inputs : def?.outputs;
  return defPool?.find((socket) => socket.id === handleId)?.kind ?? null;
}

export type ConnectionLike = {
  source: string | null;
  sourceHandle?: string | null;
  target: string | null;
  targetHandle?: string | null;
};

export type ConnectionCheck = {
  ok: boolean;
  sourceKind: string | null;
  targetKind: string | null;
  reason: string | null;
};

/**
 * Validate a prospective connection against the graph: resolve both socket
 * kinds from node data / catalog defs and compare slot types. Pure and
 * store-free, so it is unit-testable and reusable as React Flow's
 * `isValidConnection`.
 */
export function checkConnection(
  connection: ConnectionLike,
  nodes: Node[],
  defs: Map<string, NodeDef>,
): ConnectionCheck {
  const sourceNode = nodes.find((node) => node.id === connection.source);
  const targetNode = nodes.find((node) => node.id === connection.target);
  const sourceKind = resolveSocketKind(
    sourceNode,
    sourceNode ? defs.get(sourceNode.type ?? "") : undefined,
    connection.sourceHandle,
    "output",
  );
  const targetKind = resolveSocketKind(
    targetNode,
    targetNode ? defs.get(targetNode.type ?? "") : undefined,
    connection.targetHandle,
    "input",
  );
  const reason = connectionRejectionReason(sourceKind, targetKind);
  return {
    ok: reason == null,
    sourceKind,
    targetKind,
    reason,
  };
}
