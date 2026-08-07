import { expect, test } from "vitest";

import {
  createNodeAt,
  createNodeFromDef,
  defaultParamsFor,
  uniqueNodeId,
} from "../src/lib/nodeFactory";
import { useNodeRegistryStore } from "../src/store/nodeRegistry";
import { useWorkflowStore } from "../src/store/workflow";
import type { NodeDef } from "../src/ipc/schemas";

const ksamplerDef = {
  type: "builtin.ksampler",
  displayName: "KSampler",
  category: "Sampling",
  inputs: [],
  outputs: [],
  parameters: [
    { id: "seed", label: "Seed", kind: "int", default: 12345 },
    { id: "steps", label: "Steps", kind: "int", default: 30 },
    { id: "cfg", label: "CFG scale", kind: "float", default: 8.0 },
    { id: "sampler", label: "Sampler", kind: "select", default: "euler" },
    { id: "enabled", label: "Enabled", kind: "bool", default: true },
    { id: "notes", label: "Notes", kind: "string" },
  ],
} satisfies NodeDef;

/* ───── Defaults from def (F2-2) ───── */

test("defaultParamsFor keeps typed defaults, drops undefined ones", () => {
  const params = defaultParamsFor(ksamplerDef);
  expect(params).toEqual({
    seed: 12345,
    steps: 30,
    cfg: 8.0,
    sampler: "euler",
    enabled: true,
  });
  expect(params.notes).toBeUndefined();
});

test("createNodeFromDef builds a flow node with title, tone and params", () => {
  const node = createNodeFromDef(ksamplerDef, { x: 10, y: 20 }, "ksampler-1");
  expect(node.id).toBe("ksampler-1");
  expect(node.type).toBe("builtin.ksampler");
  expect(node.position).toEqual({ x: 10, y: 20 });
  expect(node.data).toMatchObject({
    title: "KSampler",
    tone: "#7928ca",
    params: { seed: 12345 },
  });
});

/* ───── Id generation ───── */

test("uniqueNodeId sanitizes type names and avoids collisions", () => {
  expect(uniqueNodeId([], "builtin.ksampler")).toBe("builtin-ksampler");
  expect(uniqueNodeId([{ id: "builtin-ksampler" }], "builtin.ksampler")).toBe("builtin-ksampler-1");
  expect(
    uniqueNodeId([{ id: "builtin-ksampler" }, { id: "builtin-ksampler-1" }], "builtin.ksampler"),
  ).toBe("builtin-ksampler-2");
});

/* ───── Imperative creation (palette / double-click / drop) ───── */

test("createNodeAt inserts from the catalog and selects the node", () => {
  useNodeRegistryStore.setState({
    defs: new Map([[ksamplerDef.type, ksamplerDef]]),
    defList: [ksamplerDef],
    status: "ready",
  });
  const before = useWorkflowStore.getState().nodes.length;
  const ok = createNodeAt("builtin.ksampler", { x: 40, y: 50 });
  expect(ok).toBe(true);
  const state = useWorkflowStore.getState();
  expect(state.nodes.length).toBe(before + 1);
  const created = state.nodes[state.nodes.length - 1];
  expect(created.type).toBe("builtin.ksampler");
  expect(created.position).toEqual({ x: 40, y: 50 });
  expect(state.selectedNode?.id).toBe(created.id);
});

test("createNodeAt returns false for unknown types without mutating", () => {
  const before = useWorkflowStore.getState().nodes.length;
  expect(createNodeAt("builtin.nope", { x: 0, y: 0 })).toBe(false);
  expect(useWorkflowStore.getState().nodes.length).toBe(before);
});
