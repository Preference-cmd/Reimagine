import { expect, test } from "bun:test";

import {
  categoryTone,
  coerceParamValue,
  formatParamValue,
  inferParamKind,
  paramSpecsFor,
  paramValueFor,
  socketKindToToken,
  type NodeLike,
} from "../src/lib/nodes";
import { artifactDisplayUrl } from "../src/lib/artifacts";
import type { NodeDef } from "../src/ipc/schemas";

/* ───── Param kind inference (F2-4) ───── */

test("infers control kinds from raw value shapes", () => {
  expect(inferParamKind("42")).toBe("int");
  expect(inferParamKind(42)).toBe("int");
  expect(inferParamKind("8.0")).toBe("float");
  expect(inferParamKind(4.5)).toBe("float");
  expect(inferParamKind(true)).toBe("bool");
  expect(inferParamKind("false")).toBe("bool");
  expect(inferParamKind("a black bear")).toBe("string");
});

test("formats typed values for display chips", () => {
  expect(formatParamValue(true)).toBe("On");
  expect(formatParamValue(false)).toBe("Off");
  expect(formatParamValue(30)).toBe("30");
  expect(formatParamValue("karras")).toBe("karras");
  expect(formatParamValue(undefined)).toBe("—");
});

test("coerces control input to the kind's canonical storage type", () => {
  expect(coerceParamValue("8", "int")).toBe(8);
  expect(coerceParamValue("8.5", "float")).toBe(8.5);
  expect(coerceParamValue("true", "bool")).toBe(true);
  expect(coerceParamValue(5, "select")).toBe("5");
});

/* ───── Socket kind mapping (F2-1) ───── */

test("maps backend socket kinds onto the four visual tokens", () => {
  expect(socketKindToToken("clip")).toEqual({ kind: "model" });
  expect(socketKindToToken("vae")).toEqual({ kind: "model" });
  expect(socketKindToToken("artifact")).toEqual({ kind: "image" });
  expect(socketKindToToken("latent")).toEqual({ kind: "latent" });
  expect(socketKindToToken("conditioning")).toEqual({ kind: "conditioning" });
  expect(socketKindToToken("mystery")).toBeNull();
});

test("maps catalog categories to header tones", () => {
  expect(categoryTone("Sampling")).toBe("#7928ca");
  expect(categoryTone("Image")).toBe("#50e3c2");
  expect(categoryTone(undefined)).toBe("#7928ca");
});

/* ───── Param spec merging (F2-1/F2-4) ───── */

const ksamplerDef = {
  type: "builtin.ksampler",
  displayName: "KSampler",
  category: "Sampling",
  inputs: [],
  outputs: [],
  parameters: [
    { id: "cfg", label: "CFG scale", kind: "float", default: 8.0 },
    { id: "sampler", label: "Sampler", kind: "select", default: "euler" },
  ],
} satisfies NodeDef;

test("catalog defs win over legacy rows; row hints fill DTO gaps", () => {
  const node: NodeLike = {
    data: {
      parameters: [
        { id: "cfg", label: "CFG scale", value: "8.0" },
        {
          id: "sampler",
          label: "Sampler",
          value: "dpm++ 2M",
          options: ["euler", "dpm++ 2M"],
        },
      ],
    },
  };
  const specs = paramSpecsFor(node, ksamplerDef);
  expect(specs).toHaveLength(2);
  expect(specs[0]).toMatchObject({ id: "cfg", kind: "float", default: 8.0 });
  expect(specs[1]).toMatchObject({
    id: "sampler",
    kind: "select",
    options: ["euler", "dpm++ 2M"],
  });
});

test("legacy rows without a catalog def get inferred kinds and row values", () => {
  const node: NodeLike = {
    data: {
      parameters: [
        { id: "steps", label: "Steps", value: "30" },
        { id: "seed", label: "Seed", value: "12345" },
      ],
    },
  };
  const specs = paramSpecsFor(node, undefined);
  expect(specs[0]).toMatchObject({ id: "steps", kind: "int", default: "30" });
  expect(specs[1]).toMatchObject({ id: "seed", kind: "int", default: "12345" });
});

test("typed data.params values win over spec defaults", () => {
  const node: NodeLike = { data: { params: { cfg: 11.5 } } };
  expect(paramValueFor(node, { id: "cfg", default: 8.0 })).toBe(11.5);
  expect(paramValueFor(node, { id: "sampler", default: "euler" })).toBe("euler");
});

/* ───── Artifact display URLs (F5-4) ───── */

test("passes data/http/asset URLs through unchanged", async () => {
  const dataUrl = "data:image/svg+xml;utf8,hello";
  expect(await artifactDisplayUrl({ path: dataUrl } as never)).toBe(dataUrl);
  expect(
    await artifactDisplayUrl({ path: "https://example.com/img.png" } as never),
  ).toBe("https://example.com/img.png");
  expect(
    await artifactDisplayUrl({ path: "asset://localhost/x.png" } as never),
  ).toBe("asset://localhost/x.png");
});

test("filesystem paths pass through when no Tauri bridge is present", async () => {
  expect(
    await artifactDisplayUrl({ path: "/tmp/out/run-1.png" } as never),
  ).toBe("/tmp/out/run-1.png");
});
