import { expect, test } from "vitest";

import {
  socketSlotType,
  isConnectionAllowed,
  connectionRejectionReason,
  resolveSocketKind,
  checkConnection,
} from "../src/lib/socketCompat";
import type { Node } from "@xyflow/react";
import type { NodeDef } from "../src/ipc/schemas";

/* ───── Slot type classification (mirror of slots.rs `SlotKind::slot_type`) ───── */

test("classifies socket kinds into the backend slot types", () => {
  expect(socketSlotType("model")).toBe("ModelHandle");
  expect(socketSlotType("model_ref")).toBe("ModelHandle");
  expect(socketSlotType("clip")).toBe("ModelHandle");
  expect(socketSlotType("vae")).toBe("ModelHandle");
  expect(socketSlotType("latent")).toBe("Tensor");
  expect(socketSlotType("conditioning")).toBe("Conditioning");
  expect(socketSlotType("image")).toBe("Artifact");
  expect(socketSlotType("artifact")).toBe("Artifact");
  expect(socketSlotType("string")).toBe("Primitive");
  expect(socketSlotType("seed")).toBe("Primitive");
  expect(socketSlotType("null")).toBe("Primitive");
  expect(socketSlotType("mystery")).toBeNull();
  expect(socketSlotType(undefined)).toBeNull();
});

/* ───── Compatibility rules (F2-3) ───── */

test("connects sockets of the same slot type", () => {
  expect(isConnectionAllowed("latent", "latent")).toBe(true);
  expect(isConnectionAllowed("model", "clip")).toBe(true);
  expect(isConnectionAllowed("image", "artifact")).toBe(true);
  expect(isConnectionAllowed("conditioning", "conditioning")).toBe(true);
  expect(isConnectionAllowed("string", "seed")).toBe(true);
});

test("rejects cross-type connections", () => {
  expect(isConnectionAllowed("model", "latent")).toBe(false);
  expect(isConnectionAllowed("latent", "image")).toBe(false);
  expect(isConnectionAllowed("conditioning", "image")).toBe(false);
  expect(isConnectionAllowed("image", "conditioning")).toBe(false);
  expect(isConnectionAllowed("clip", "conditioning")).toBe(false);
});

test("unknown kinds stay connectable (forward compatibility)", () => {
  expect(isConnectionAllowed("mystery_kind", "latent")).toBe(true);
  expect(isConnectionAllowed("model", "another_mystery")).toBe(true);
  expect(isConnectionAllowed(null, "latent")).toBe(true);
});

test("rejection reason is null when allowed, descriptive otherwise", () => {
  expect(connectionRejectionReason("latent", "latent")).toBeNull();
  expect(connectionRejectionReason("model", "latent")).toContain("Cannot connect model → latent");
  expect(connectionRejectionReason(null, null)).toBeNull();
});

/* ───── Socket kind resolution ───── */

test("resolves kinds from embedded node data first", () => {
  const node = {
    id: "a",
    data: {
      inputs: [{ id: "model", kind: "model", label: "model" }],
      outputs: [{ id: "out", kind: "image", label: "out" }],
    },
  } as unknown as Node;
  expect(resolveSocketKind(node, undefined, "model", "input")).toBe("model");
  expect(resolveSocketKind(node, undefined, "out", "output")).toBe("image");
  expect(resolveSocketKind(node, undefined, "nope", "input")).toBeNull();
});

test("falls back to catalog defs for schema-driven nodes", () => {
  const node = { id: "b", data: { params: {} } } as unknown as Node;
  const def = {
    type: "builtin.ksampler",
    displayName: "KSampler",
    category: "Sampling",
    inputs: [
      { id: "model", kind: "model", label: "model" },
      { id: "latent", kind: "latent", label: "latent" },
    ],
    outputs: [{ id: "latent", kind: "latent", label: "latent" }],
    parameters: [],
  } satisfies NodeDef;
  expect(resolveSocketKind(node, def, "model", "input")).toBe("model");
  expect(resolveSocketKind(node, def, "latent", "input")).toBe("latent");
  expect(resolveSocketKind(node, def, "latent", "output")).toBe("latent");
  expect(resolveSocketKind(node, def, "missing", "input")).toBeNull();
});

/* ───── Full connection checks ───── */

const samplerDef = {
  type: "builtin.ksampler",
  displayName: "KSampler",
  category: "Sampling",
  inputs: [
    { id: "model", kind: "model", label: "model" },
    { id: "positive", kind: "conditioning", label: "positive" },
    { id: "latent", kind: "latent", label: "latent" },
  ],
  outputs: [{ id: "latent", kind: "latent", label: "latent" }],
  parameters: [],
} satisfies NodeDef;

const loaderDef = {
  type: "builtin.checkpoint_loader",
  displayName: "Checkpoint Loader",
  category: "Model",
  inputs: [],
  outputs: [
    { id: "model", kind: "model", label: "model" },
    { id: "clip", kind: "clip", label: "clip" },
    { id: "vae", kind: "vae", label: "vae" },
  ],
  parameters: [],
} satisfies NodeDef;

const nodes = [
  { id: "loader", type: "builtin.checkpoint_loader", data: { params: {} } },
  { id: "sampler", type: "builtin.ksampler", data: { params: {} } },
] as unknown as Node[];

const defs = new Map<string, NodeDef>([
  ["builtin.checkpoint_loader", loaderDef],
  ["builtin.ksampler", samplerDef],
]);

test("checkConnection accepts model → model", () => {
  const result = checkConnection(
    { source: "loader", sourceHandle: "model", target: "sampler", targetHandle: "model" },
    nodes,
    defs,
  );
  expect(result.ok).toBe(true);
  expect(result.sourceKind).toBe("model");
  expect(result.targetKind).toBe("model");
});

test("checkConnection rejects model → latent with a reason", () => {
  const result = checkConnection(
    { source: "loader", sourceHandle: "model", target: "sampler", targetHandle: "latent" },
    nodes,
    defs,
  );
  expect(result.ok).toBe(false);
  expect(result.sourceKind).toBe("model");
  expect(result.targetKind).toBe("latent");
  expect(result.reason).toContain("Cannot connect model → latent");
});

test("checkConnection treats unknown handles as compatible", () => {
  const result = checkConnection(
    { source: "sampler", sourceHandle: "latent", target: "loader", targetHandle: "ghost" },
    nodes,
    defs,
  );
  expect(result.ok).toBe(true);
  expect(result.targetKind).toBeNull();
});
