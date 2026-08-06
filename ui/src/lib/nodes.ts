import type { NodeDef, ParamKind, ParamSpec, SocketSpec } from "@/ipc/schemas";
import type { SocketSlot, ParamRow } from "@/components/canvas/BaseNode";

/**
 * Pure helpers shared by GenericNode, PropertiesPanel and the catalog
 * registry. Everything here is node-catalog-aware but React-free, so the
 * param-kind inference and spec merging stay unit-testable.
 *
 * Helpers accept a `NodeLike` (anything carrying `.data`) so both real
 * React Flow nodes and bare `{ data }` objects work.
 */

/** Typed parameter values stored on node data (`data.params`). */
export type ParamValue = string | number | boolean;

/** A catalog param spec, enriched with legacy-row fallbacks. */
export type ParamSpecLike = ParamSpec & {
  /** Select options — the backend DTO serializes them when the node
      declares constraints; legacy row hints fill in when absent. */
  options?: string[];
};

/** Legacy demo row with optional control hints (kind/options). */
export type ParamRowLike = ParamRow & {
  kind?: ParamKind;
  options?: string[];
  min?: number;
  max?: number;
};

/** Anything with a `data` payload (React Flow `Node` fits this). */
export type NodeLike = { data?: unknown };

/* ───── Category → tone ───── */

const CATEGORY_TONES: Record<string, string> = {
  Model: "#f5a623",
  Conditioning: "#50e3c2",
  Latent: "#7928ca",
  Sampling: "#7928ca",
  Image: "#50e3c2",
  VAE: "#0070f3",
  Input: "#f5a623",
  Output: "#50e3c2",
};

/** Category name → header tone hex (with a sane default for unknown ones). */
export function categoryTone(category: string | undefined): string {
  return (category && CATEGORY_TONES[category]) || "#7928ca";
}

/* ───── Socket kind mapping ───── */

/**
 * Map an open backend socket kind onto the four visual socket tokens
 * (SOCKET_COLORS). Model-ish handles (clip/vae/model_ref) share the
 * model route; artifact handles share the image route. Unknown kinds
 * yield `null` — the caller should hide the socket rather than guess.
 */
export function socketKindToToken(
  kind: string,
): { kind: SocketSlot["kind"]; dotColor?: string } | null {
  switch (kind) {
    case "model":
    case "model_ref":
    case "clip":
    case "vae":
      return { kind: "model" };
    case "conditioning":
      return { kind: "conditioning" };
    case "latent":
      return { kind: "latent" };
    case "image":
    case "artifact":
      return { kind: "image" };
    default:
      return null;
  }
}

/** Sockets (left column) for a node: legacy data first, catalog defs as fallback. */
export function inputSlotsFor(node: NodeLike, def?: NodeDef): SocketSlot[] {
  const data = (node.data ?? {}) as { inputs?: unknown };
  if (Array.isArray(data.inputs)) {
    return data.inputs.filter(isSocketSlot);
  }
  return (def?.inputs ?? []).flatMap(socketSpecToSlot);
}

/** Sockets (right column) for a node: legacy data first, catalog defs as fallback. */
export function outputSlotsFor(node: NodeLike, def?: NodeDef): SocketSlot[] {
  const data = (node.data ?? {}) as { outputs?: unknown };
  if (Array.isArray(data.outputs)) {
    return data.outputs.filter(isSocketSlot);
  }
  return (def?.outputs ?? []).flatMap(socketSpecToSlot);
}

/** Catalog socket → visual slot; unmappable kinds are skipped. */
function socketSpecToSlot(spec: SocketSpec): SocketSlot[] {
  const mapped = socketKindToToken(spec.kind);
  if (!mapped) return [];
  return [{ id: spec.id, label: spec.label, ...mapped }];
}

function isSocketSlot(value: unknown): value is SocketSlot {
  if (typeof value !== "object" || value == null) return false;
  const slot = value as Record<string, unknown>;
  return (
    typeof slot.id === "string" &&
    typeof slot.label === "string" &&
    typeof slot.kind === "string"
  );
}

/* ───── Parameter specs ───── */

/**
 * Effective editable parameter list for a node.
 *
 * Catalog-driven nodes use the backend `ParamSpec[]` (authoritative
 * kinds and constraint data); legacy demo rows are merged in only to
 * fill gaps the DTO leaves unset. Nodes without a catalog entry fall
 * back to their legacy rows with kinds inferred from the current value.
 */
export function paramSpecsFor(node: NodeLike, def?: NodeDef): ParamSpecLike[] {
  const rows = readParamRows(node.data);
  const defParams = def?.parameters ?? [];

  if (defParams.length > 0) {
    return defParams.map((spec) => {
      const row = rows.find((r) => r.id === spec.id);
      return {
        ...spec,
        options:
          (spec.options?.length ?? 0) > 0
            ? spec.options
            : row?.options ?? undefined,
        default: spec.default ?? row?.value,
        min: spec.min ?? row?.min,
        max: spec.max ?? row?.max,
      };
    });
  }

  return rows.map((row) => ({
    id: row.id,
    label: row.label,
    kind: row.kind ?? inferParamKind(row.value),
    default: row.value,
    options: row.options,
    min: row.min,
    max: row.max,
  }));
}

/** Current value for a param spec: typed node data → spec default. */
export function paramValueFor(
  node: NodeLike,
  spec: { id: string; default?: unknown },
): ParamValue | undefined {
  const data = (node.data ?? {}) as { params?: Record<string, unknown> };
  const own = data.params?.[spec.id];
  if (own !== undefined) {
    return normalizeParamValue(own);
  }
  return normalizeParamValue(spec.default);
}

/** Coerce a raw JSON value to a frontend ParamValue (or undefined). */
export function normalizeParamValue(value: unknown): ParamValue | undefined {
  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value;
  }
  return undefined;
}

/** Infer a control kind from a value's shape (legacy rows, no catalog). */
export function inferParamKind(value: unknown): ParamKind {
  if (typeof value === "number") {
    return Number.isInteger(value) ? "int" : "float";
  }
  if (typeof value === "boolean") return "bool";
  if (typeof value === "string") {
    if (/^-?\d+$/.test(value)) return "int";
    if (/^-?\d*\.\d+$/.test(value)) return "float";
    if (value === "true" || value === "false") return "bool";
  }
  return "string";
}

/** Render a typed value for display chips. */
export function formatParamValue(value: unknown): string {
  if (typeof value === "boolean") return value ? "On" : "Off";
  if (value === null || value === undefined) return "—";
  return String(value);
}

/** Coerce a control value to the kind's canonical type for storage. */
export function coerceParamValue(
  value: string | number | boolean,
  kind: ParamKind,
): ParamValue {
  switch (kind) {
    case "int": {
      const n = typeof value === "number" ? value : Number(value);
      return Number.isFinite(n) ? Math.trunc(n) : 0;
    }
    case "float": {
      const n = typeof value === "number" ? value : Number(value);
      return Number.isFinite(n) ? n : 0;
    }
    case "bool":
      return typeof value === "boolean" ? value : value === "true";
    default:
      return String(value);
  }
}

/* ───── Local ───── */

function readParamRows(data: unknown): ParamRowLike[] {
  if (typeof data !== "object" || data == null) return [];
  const rows = (data as { parameters?: unknown }).parameters;
  if (!Array.isArray(rows)) return [];
  return rows.flatMap((item) => {
    if (typeof item !== "object" || item == null) return [];
    const record = item as Record<string, unknown>;
    const id = typeof record.id === "string" ? record.id : "";
    const label = typeof record.label === "string" ? record.label : "";
    const value = record.value;
    if (!id) return [];
    const row: ParamRowLike = {
      id,
      label,
      value: String(value ?? ""),
    };
  if (typeof record.kind === "string") row.kind = record.kind as ParamKind;
  if (Array.isArray(record.options) && record.options.every((o) => typeof o === "string")) {
    row.options = record.options;
  }
  if (typeof record.min === "number") row.min = record.min;
  if (typeof record.max === "number") row.max = record.max;
  return [row];
  });
}
