import { opInvoke } from "@/ipc";
import {
  AssetSummaryListSchema,
  CollectionSummaryListSchema,
  DynamicWorkflowRunListSchema,
  DynamicWorkflowSummaryListSchema,
  type AssetSummary,
  type BoardItem,
  type BoardItemKind,
  type CollectionSummary,
  type DynamicWorkflowRun,
  type DynamicWorkflowSummary,
  type RunState,
} from "@/ipc/schemas";

/** Row kinds in the library table, in display order. */
export const LIBRARY_ROW_TYPES = ["collection", "workflow", "run", "asset"] as const;
export type LibraryRowType = (typeof LIBRARY_ROW_TYPES)[number];

/** Chip filters: "all" plus one per row kind. */
export const LIBRARY_FILTERS = ["all", ...LIBRARY_ROW_TYPES] as const;
export type LibraryFilter = (typeof LIBRARY_FILTERS)[number];

export type LibraryRow = {
  type: LibraryRowType;
  id: string;
  name: string;
  detail: string;
};

export type LibraryRowLabels = {
  collectionDetail: (slotCount: number, version: number) => string;
  workflowDetail: (stepCount: number, version: number) => string;
  runState: (state: RunState) => string;
};

export function isLibraryFilter(value: string | null | undefined): value is LibraryFilter {
  return typeof value === "string" && (LIBRARY_FILTERS as readonly string[]).includes(value);
}

/** Human-readable byte size, one decimal below 100 and none above. */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  if (unitIndex === 0) return `${Math.round(value)} B`;
  const rounded = value >= 100 ? Math.round(value) : Math.round(value * 10) / 10;
  return `${rounded} ${units[unitIndex]}`;
}

/**
 * Flattens the four list reads into table rows. Runs join their workflow
 * name by id and fall back to the raw id when the workflow is gone; a short
 * run id suffix keeps several runs of the same workflow apart.
 * Rows are sorted by type order, then by name.
 */
export function buildLibraryRows(
  input: {
    collections: CollectionSummary[];
    workflows: DynamicWorkflowSummary[];
    runs: DynamicWorkflowRun[];
    assets: AssetSummary[];
  },
  labels: LibraryRowLabels,
): LibraryRow[] {
  const workflowNames = new Map(input.workflows.map((workflow) => [workflow.id, workflow.name]));

  const rows: LibraryRow[] = [
    ...input.collections.map((collection) => ({
      type: "collection" as const,
      id: collection.id,
      name: collection.name,
      detail: labels.collectionDetail(collection.slot_count, collection.version),
    })),
    ...input.workflows.map((workflow) => ({
      type: "workflow" as const,
      id: workflow.id,
      name: workflow.name,
      detail: labels.workflowDetail(workflow.step_count, workflow.version),
    })),
    ...input.runs.map((run) => {
      const succeeded = run.items.filter((item) => item.state === "succeeded").length;
      const counter = run.items.length > 0 ? ` · ${succeeded}/${run.items.length}` : "";
      const workflowName = workflowNames.get(run.dynamic_workflow_id) ?? run.dynamic_workflow_id;
      return {
        type: "run" as const,
        id: run.id,
        name: `${workflowName} #${run.id.slice(-6)}`,
        detail: `${labels.runState(run.state)}${counter}`,
      };
    }),
    ...input.assets.map((asset) => ({
      type: "asset" as const,
      id: asset.id,
      name: asset.id,
      detail: `${asset.media_type} · ${formatBytes(asset.size)}`,
    })),
  ];

  return rows.sort((a, b) => {
    const byType = LIBRARY_ROW_TYPES.indexOf(a.type) - LIBRARY_ROW_TYPES.indexOf(b.type);
    return byType !== 0 ? byType : a.name.localeCompare(b.name);
  });
}

/** Chip counts over every row, independent of the current search. */
export function countLibraryRows(rows: LibraryRow[]): Record<LibraryFilter, number> {
  const counts: Record<LibraryFilter, number> = {
    all: rows.length,
    collection: 0,
    workflow: 0,
    run: 0,
    asset: 0,
  };
  for (const row of rows) counts[row.type] += 1;
  return counts;
}

/** Filters by type chip and a trimmed, case-insensitive name/detail search. */
export function filterLibraryRows(
  rows: LibraryRow[],
  filter: LibraryFilter,
  query: string,
): LibraryRow[] {
  const needle = query.trim().toLowerCase();
  return rows.filter((row) => {
    if (filter !== "all" && row.type !== filter) return false;
    if (!needle) return true;
    return row.name.toLowerCase().includes(needle) || row.detail.toLowerCase().includes(needle);
  });
}

/* ───── Library rows on the board ───── */

/** Card sizes used when a library row is placed, in flow px. */
export const LIBRARY_PLACEMENT_SIZES: Record<LibraryRowType, { width: number; height: number }> = {
  collection: { width: 300, height: 200 },
  workflow: { width: 280, height: 160 },
  run: { width: 280, height: 160 },
  asset: { width: 240, height: 160 },
};

/**
 * The board item a library row stands for. Imported files reuse `asset_ref`
 * with the asset id in place of an artifact id; there is no file kind.
 */
export function libraryRowBoardKind(row: Pick<LibraryRow, "type" | "id">): BoardItemKind {
  switch (row.type) {
    case "collection":
      return { type: "collection", collection_id: row.id };
    case "workflow":
      return { type: "workflow_ref", workflow_id: row.id };
    case "run":
      return { type: "run_ref", run_id: row.id };
    case "asset":
      return { type: "asset_ref", artifact_id: row.id };
  }
}

/** True when `item` is the board copy of `row` (same kind and same id field). */
export function boardItemMatchesRow(
  item: BoardItem,
  row: Pick<LibraryRow, "type" | "id">,
): boolean {
  const kind = item.kind as { type: string; [field: string]: unknown };
  switch (row.type) {
    case "collection":
      return kind.type === "collection" && kind.collection_id === row.id;
    case "workflow":
      return kind.type === "workflow_ref" && kind.workflow_id === row.id;
    case "run":
      return kind.type === "run_ref" && kind.run_id === row.id;
    case "asset":
      return kind.type === "asset_ref" && kind.artifact_id === row.id;
  }
}

/** Top-left position that centers a card of `size` on the given flow point. */
export function centeredPosition(
  center: { x: number; y: number },
  size: { width: number; height: number },
): { x: number; y: number } {
  return { x: center.x - size.width / 2, y: center.y - size.height / 2 };
}

/* ───── Typed library reads (op_invoke + output parsing) ───── */

/** Collections in a project (id, name, version, slot count). */
export async function listCollections(projectId: string): Promise<CollectionSummary[]> {
  const result = await opInvoke(projectId, "collection.list", {});
  return CollectionSummaryListSchema.parse(result.output).collections;
}

/** Dynamic workflow documents in a project (id, name, version, step count). */
export async function listDynamicWorkflows(projectId: string): Promise<DynamicWorkflowSummary[]> {
  const result = await opInvoke(projectId, "dynamic_workflow.list", {});
  return DynamicWorkflowSummaryListSchema.parse(result.output).workflows;
}

/** Assets imported into a project, newest first. */
export async function listAssets(projectId: string): Promise<AssetSummary[]> {
  const result = await opInvoke(projectId, "asset.list", {});
  return AssetSummaryListSchema.parse(result.output).assets;
}

/** Dynamic workflow runs, newest first; omit `collectionId` for project-wide. */
export async function listRuns(
  projectId: string,
  limit = 50,
  collectionId?: string,
): Promise<DynamicWorkflowRun[]> {
  const input = collectionId ? { collection_id: collectionId, limit } : { limit };
  const result = await opInvoke(projectId, "dynamic_workflow.list_runs", input);
  return DynamicWorkflowRunListSchema.parse(result.output).runs;
}
