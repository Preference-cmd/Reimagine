import { expect, test } from "bun:test";

import { fuzzyScore, filterEntries, type PaletteEntry } from "../src/lib/palette";

/* ───── Fuzzy scoring (F3-3) ───── */

test("scores prefix matches highest, then contains, then subsequence", () => {
  const text = "ksample";
  const prefix = fuzzyScore("ksam", text);
  const contains = fuzzyScore("samp", text);
  const subsequence = fuzzyScore("kse", text);
  expect(prefix).toBeGreaterThan(contains);
  expect(contains).toBeGreaterThan(subsequence);
  expect(subsequence).toBeGreaterThan(0);
});

test("non-subsequences score zero; matching is case-insensitive", () => {
  expect(fuzzyScore("zzz", "ksampler")).toBe(0);
  expect(fuzzyScore("KSAMPLER", "ksampler")).toBeGreaterThan(0);
  expect(fuzzyScore("", "anything")).toBe(0);
  expect(fuzzyScore("   ", "anything")).toBe(0);
});

test("word-boundary starts beat mid-word matches", () => {
  expect(fuzzyScore("lat", "lat_image")).toBeGreaterThan(
    fuzzyScore("lat", "collateral"),
  );
});

/* ───── Entry filtering ───── */

const entries: PaletteEntry[] = [
  { id: "run-workflow", label: "Run Workflow", keywords: ["execute", "start"] },
  { id: "save", label: "Save", keywords: ["persist"] },
  { id: "add-node", label: "Add Node…", keywords: ["create"] },
  {
    id: "node-latent",
    label: "Empty Latent Image",
    hint: "builtin.empty_latent_image",
    keywords: ["Latent"],
  },
];

test("empty query keeps catalog order", () => {
  expect(filterEntries("", entries).map((e) => e.id)).toEqual([
    "run-workflow",
    "save",
    "add-node",
    "node-latent",
  ]);
});

test("prefix matches rank first, keyword subsequences extend the search", () => {
  const result = filterEntries("sa", entries);
  expect(result.map((e) => e.id)).toEqual(["save", "run-workflow"]);
  expect(result.map((e) => e.id)).not.toContain("add-node");
});

test("keywords and hints extend the search surface", () => {
  expect(filterEntries("execute", entries).map((e) => e.id)).toEqual([
    "run-workflow",
  ]);
  expect(filterEntries("empty_latent", entries).map((e) => e.id)).toEqual([
    "node-latent",
  ]);
  expect(filterEntries("persist", entries).map((e) => e.id)).toEqual(["save"]);
});

test("no matches yields an empty list", () => {
  expect(filterEntries("zzzz", entries)).toEqual([]);
});
