import { useEffect, useMemo, useState, type ComponentType } from "react";
import {
  Boxes,
  CircleAlert,
  FileText,
  Layers,
  Library,
  Loader2,
  LocateFixed,
  Play,
  Plus,
  Search,
  SearchX,
  Upload,
} from "lucide-react";
import { Link, useNavigate, useSearch } from "@tanstack/react-router";
import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import * as m from "$paraglide/messages";
import { ICON_STROKE } from "@/design/tokens";
import { cn } from "@/lib/utils";
import { useReducedMotion } from "@/hooks/useReducedMotion";
import { useLibraryRows, libraryQueryKeys } from "@/hooks/useLibrary";
import { useProjectStore } from "@/store/project";
import { useBoardStore } from "@/store/board";
import { useUIStore } from "@/store/uiStore";
import { isTauri, opInvoke } from "@/ipc";
import { buildAddItemCommand, createBoardBatch } from "@/board/batch";
import {
  boardItemMatchesRow,
  centeredPosition,
  filterLibraryRows,
  isLibraryFilter,
  LIBRARY_PLACEMENT_SIZES,
  libraryRowBoardKind,
  type LibraryFilter,
  type LibraryRow,
  type LibraryRowType,
} from "@/lib/library";

type RowIcon = ComponentType<{ className?: string; strokeWidth?: number }>;

const ROW_ICONS: Record<LibraryRowType, RowIcon> = {
  collection: Layers,
  workflow: Boxes,
  run: Play,
  asset: FileText,
};

function rowTypeLabel(type: LibraryRowType): string {
  switch (type) {
    case "collection":
      return m["library.typeCollection"]();
    case "workflow":
      return m["library.typeWorkflow"]();
    case "run":
      return m["library.typeRun"]();
    case "asset":
      return m["library.typeAsset"]();
  }
}

const PLACE_BUTTON =
  "inline-flex h-7 cursor-pointer items-center gap-1.5 rounded-full bg-surface px-2.5 text-caption font-medium whitespace-nowrap text-fg shadow-[inset_0_0_0_1px_var(--color-hair-strong),var(--shadow-hi)] transition-colors hover:bg-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/30 disabled:cursor-not-allowed disabled:opacity-50";

const SHOW_BUTTON =
  "inline-flex h-7 cursor-pointer items-center gap-1.5 rounded-full px-2.5 text-caption font-medium whitespace-nowrap text-fg2 transition-colors hover:bg-app hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/30";

function LibraryRowView({ row, projectId }: { row: LibraryRow; projectId: string }) {
  const Icon = ROW_ICONS[row.type];
  const navigate = useNavigate();
  const [placing, setPlacing] = useState(false);
  // A boolean selector over the board items: the row flips as soon as a
  // snapshot carrying this item lands, without subscribing to unrelated state.
  const onBoard = useBoardStore((state) =>
    state.items.some((item) => boardItemMatchesRow(item, row)),
  );

  function handleShow() {
    const item = useBoardStore
      .getState()
      .items.find((candidate) => boardItemMatchesRow(candidate, row));
    if (item) useUIStore.getState().requestCanvasFocus(item.id);
    void navigate({ to: "/" });
  }

  async function handlePlace() {
    if (placing) return;
    setPlacing(true);
    try {
      let board = useBoardStore.getState();
      if (board.id === "") {
        // Without a snapshot the version is 0 and the board may already hold
        // this item, so load first and re-check before adding a second copy.
        await board.load(projectId);
        board = useBoardStore.getState();
        if (board.items.some((item) => boardItemMatchesRow(item, row))) return;
      }

      const center = useUIStore.getState().lastCanvasCenter ?? { x: 0, y: 0 };
      const size = LIBRARY_PLACEMENT_SIZES[row.type];
      const command = buildAddItemCommand({
        kind: libraryRowBoardKind(row),
        position: centeredPosition(center, size),
        size,
      });
      const batch = createBoardBatch(board.version, [command]);
      const result = await useBoardStore.getState().apply(projectId, batch);

      if (result.status === "rejected") {
        const firstDiag = result.diagnostics?.[0] as { message?: string } | undefined;
        toast.error(firstDiag?.message ?? m["library.placeFailed"]());
        await useBoardStore
          .getState()
          .load(projectId)
          .catch(() => false);
        return;
      }

      // The host accepted the command, so refresh the snapshot now instead of
      // waiting for the board event; the row flips without a manual reload.
      await useBoardStore
        .getState()
        .load(projectId)
        .catch(() => false);
      toast.success(m["library.placed"]());
    } catch {
      toast.error(m["library.placeFailed"]());
    } finally {
      setPlacing(false);
    }
  }

  return (
    <tr className="group" data-testid="library-row">
      <td className="h-[46px] rounded-l-xs border-b border-hair-soft px-3 transition-colors group-hover:bg-hover">
        <span className="flex min-w-0 items-center gap-2.5 font-medium text-fg">
          <Icon
            className="size-3.5 shrink-0 text-fg3"
            strokeWidth={ICON_STROKE}
            aria-hidden="true"
          />
          {row.type === "collection" ? (
            <Link
              to="/collections/$id"
              params={{ id: row.id }}
              className="min-w-0 truncate rounded-xs hover:underline focus-visible:ring-2 focus-visible:ring-accent/30 focus-visible:outline-none"
              title={row.name}
              data-testid="library-collection-link"
            >
              {row.name}
            </Link>
          ) : (
            <span className="truncate" title={row.name}>
              {row.name}
            </span>
          )}
        </span>
      </td>
      <td className="h-[46px] truncate border-b border-hair-soft px-3 text-fg2 transition-colors group-hover:bg-hover">
        {rowTypeLabel(row.type)}
      </td>
      <td
        className="h-[46px] truncate border-b border-hair-soft px-3 text-fg2 transition-colors group-hover:bg-hover"
        title={row.detail}
      >
        {row.detail}
      </td>
      <td className="h-[46px] rounded-r-xs border-b border-hair-soft px-3 text-right transition-colors group-hover:bg-hover">
        {onBoard ? (
          <button
            type="button"
            onClick={handleShow}
            data-testid="library-row-action"
            data-action="show"
            className={SHOW_BUTTON}
          >
            <LocateFixed className="size-3.5" strokeWidth={ICON_STROKE} aria-hidden="true" />
            {m["library.showInCanvas"]()}
          </button>
        ) : (
          <button
            type="button"
            onClick={() => void handlePlace()}
            disabled={placing}
            data-testid="library-row-action"
            data-action="place"
            className={PLACE_BUTTON}
          >
            <Plus className="size-3.5" strokeWidth={ICON_STROKE} aria-hidden="true" />
            {m["library.placeOnCanvas"]()}
          </button>
        )}
      </td>
    </tr>
  );
}

function LibraryTable({ rows, projectId }: { rows: LibraryRow[]; projectId: string }) {
  return (
    <div className="min-w-0 overflow-x-auto">
      <table
        className="w-full table-fixed border-separate border-spacing-0 text-ui"
        data-testid="library-table"
      >
        <thead>
          <tr className="text-left text-caption text-fg2">
            <th
              scope="col"
              className="w-[41%] border-b border-hair px-3 pb-2 font-normal whitespace-nowrap"
            >
              {m["library.columnName"]()}
            </th>
            <th
              scope="col"
              className="w-[18%] border-b border-hair px-3 pb-2 font-normal whitespace-nowrap"
            >
              {m["library.columnType"]()}
            </th>
            <th
              scope="col"
              className="border-b border-hair px-3 pb-2 font-normal whitespace-nowrap"
            >
              {m["library.columnDetails"]()}
            </th>
            <th
              scope="col"
              className="w-[158px] border-b border-hair px-3 pb-2 text-right font-normal whitespace-nowrap"
            >
              <span className="sr-only">{m["library.actions"]()}</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <LibraryRowView key={`${row.type}:${row.id}`} row={row} projectId={projectId} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * Library mode: the workspace's collections, dynamic workflows, runs and
 * imported files in one filterable table. Each row can be placed on the canvas
 * at the last visible center, or focused there when it is already on it.
 */
export function LibraryView() {
  const panelOpen = useUIStore((state) => state.sessionPanelOpen);
  const reducedMotion = useReducedMotion();
  const projectId = useProjectStore((state) => state.activeProjectId);
  const navigate = useNavigate();
  const search = useSearch({ from: "/library" });
  const queryClient = useQueryClient();

  const urlFilter: LibraryFilter = isLibraryFilter(search.type) ? search.type : "all";
  const [filter, setFilter] = useState<LibraryFilter>(urlFilter);
  const [query, setQuery] = useState("");
  const [importing, setImporting] = useState(false);
  const tauri = isTauri();

  // The sidebar and topbar run shortcuts land on /library?type=run; keep the
  // chip in step when the param changes after mount.
  useEffect(() => {
    setFilter(urlFilter);
  }, [urlFilter]);

  const { rows, counts, isLoading, error, refetch } = useLibraryRows(projectId);
  const visibleRows = useMemo(() => filterLibraryRows(rows, filter, query), [rows, filter, query]);

  const chips: Array<{ value: LibraryFilter; label: string; count: number }> = [
    { value: "all", label: m["library.filterAll"](), count: counts.all },
    { value: "collection", label: m["library.filterCollections"](), count: counts.collection },
    { value: "workflow", label: m["library.filterWorkflows"](), count: counts.workflow },
    { value: "run", label: m["library.filterRuns"](), count: counts.run },
    { value: "asset", label: m["library.filterFiles"](), count: counts.asset },
  ];

  function selectFilter(next: LibraryFilter) {
    setFilter(next);
    void navigate({
      to: "/library",
      search: { type: next === "all" ? undefined : next },
      replace: true,
    });
  }

  async function handleImport() {
    if (!tauri || importing) return;
    setImporting(true);
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selection = await open({ multiple: true, directory: false });
      if (!selection) return;
      const paths = Array.isArray(selection) ? selection : [selection];
      if (paths.length === 0) return;

      const result = await opInvoke(projectId, "asset.import", { paths });
      const imported = (result.output as { assets?: unknown } | null)?.assets;
      if (!Array.isArray(imported) || imported.length === 0) {
        throw new Error("no assets imported");
      }
      await queryClient.invalidateQueries({ queryKey: libraryQueryKeys.assets(projectId) });
      toast.success(m["library.imported"]());
    } catch {
      toast.error(m["library.importFailed"]());
    } finally {
      setImporting(false);
    }
  }

  const importReason = !tauri ? m["library.importUnavailable"]() : null;
  const importLabel = importReason
    ? `${m["library.importFiles"]()} — ${importReason}`
    : m["library.importFiles"]();

  return (
    <div
      className={cn(
        "flex h-full min-h-0 flex-col overflow-y-auto bg-surface px-6 pt-5 pb-12",
        !reducedMotion && "transition-[padding] duration-300 ease-ui",
      )}
      style={
        panelOpen
          ? { paddingRight: "calc(var(--spacing-panel) + var(--spacing-panel-gap) * 2)" }
          : undefined
      }
    >
      <header className="flex shrink-0 items-start justify-between gap-4">
        <div>
          <h1 className="text-headline-md font-semibold tracking-[-0.02em] text-fg">
            {m["library.title"]()}
          </h1>
          <p className="mt-1 text-ui text-fg2">{m["library.subtitle"]()}</p>
        </div>
        <button
          type="button"
          onClick={() => void handleImport()}
          disabled={!tauri || importing}
          aria-label={importing ? m["library.importing"]() : importLabel}
          title={importReason ?? undefined}
          data-testid="library-import"
          className="mt-0.5 inline-flex h-8 shrink-0 cursor-pointer items-center gap-1.5 rounded-full bg-surface px-3 text-caption font-medium text-fg shadow-[inset_0_0_0_1px_var(--color-hair-strong),var(--shadow-hi)] transition-colors hover:bg-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/30 disabled:cursor-not-allowed disabled:opacity-45 disabled:hover:bg-surface"
        >
          {importing ? (
            <Loader2
              className="size-3.5 animate-spin"
              strokeWidth={ICON_STROKE}
              aria-hidden="true"
            />
          ) : (
            <Upload className="size-3.5" strokeWidth={ICON_STROKE} aria-hidden="true" />
          )}
          {importing ? m["library.importing"]() : m["library.importFiles"]()}
        </button>
      </header>

      <div className="mt-4 flex shrink-0 flex-wrap items-center gap-2">
        <label className="flex h-8 w-60 items-center gap-2 rounded-full px-3 text-fg3 shadow-[inset_0_0_0_1px_var(--color-hair-strong)] transition-shadow focus-within:text-fg2 focus-within:shadow-[inset_0_0_0_1px_var(--color-accent-line),0_0_0_4px_var(--color-accent-tint)]">
          <Search className="size-3.5 shrink-0" strokeWidth={ICON_STROKE} aria-hidden="true" />
          <input
            type="text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            aria-label={m["library.searchLabel"]()}
            placeholder={m["library.searchPlaceholder"]()}
            spellCheck={false}
            data-testid="library-search"
            className="min-w-0 flex-1 border-0 bg-transparent text-ui text-fg outline-none placeholder:text-fg2"
          />
        </label>

        <div
          role="group"
          aria-label={m["library.filterGroup"]()}
          className="flex flex-wrap items-center gap-1.5"
        >
          {chips.map((chip) => {
            const selected = filter === chip.value;
            return (
              <button
                key={chip.value}
                type="button"
                aria-pressed={selected}
                data-testid={`library-chip-${chip.value}`}
                onClick={() => selectFilter(chip.value)}
                className={cn(
                  "inline-flex h-7 cursor-pointer items-center gap-1.5 rounded-full px-2.5 text-caption transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/30",
                  selected
                    ? "bg-accent-tint font-medium text-accent shadow-[inset_0_0_0_1px_var(--color-accent-line)]"
                    : "text-fg2 shadow-[inset_0_0_0_1px_var(--color-hair)] hover:bg-hover hover:text-fg",
                )}
              >
                {chip.label}
                <span className="font-mono text-[11px] tabular-nums opacity-65">{chip.count}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="mt-4 flex min-h-0 flex-1 flex-col">
        {isLoading ? (
          <div
            className="flex flex-1 items-center justify-center gap-2 py-16 text-ui text-fg2"
            role="status"
            data-testid="library-loading"
          >
            <Loader2 className="size-4 animate-spin" strokeWidth={ICON_STROKE} aria-hidden="true" />
            {m["library.loading"]()}
          </div>
        ) : error ? (
          <div
            className="flex flex-1 flex-col items-center justify-center py-16 text-center"
            role="alert"
            data-testid="library-error"
          >
            <span className="grid size-10 place-items-center rounded-xl bg-tile text-err shadow-[var(--shadow-hi)]">
              <CircleAlert className="size-4" strokeWidth={ICON_STROKE} aria-hidden="true" />
            </span>
            <h2 className="mt-4 text-ui font-medium text-fg">{m["library.errorTitle"]()}</h2>
            <p className="mt-1.5 max-w-[24rem] text-ui leading-6 text-fg2">
              {m["library.errorDescription"]()}
            </p>
            <p className="mt-2 max-w-[26rem] truncate font-mono text-caption text-fg2">
              {error.message}
            </p>
            <button
              type="button"
              onClick={() => void refetch()}
              data-testid="library-retry"
              className="mt-4 inline-flex h-8 cursor-pointer items-center rounded-full bg-surface px-3 text-caption font-medium text-fg shadow-[inset_0_0_0_1px_var(--color-hair-strong),var(--shadow-hi)] transition-colors hover:bg-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/30"
            >
              {m["library.retry"]()}
            </button>
          </div>
        ) : rows.length === 0 ? (
          <div className="flex flex-1 flex-col items-center justify-center py-16 text-center">
            <span className="grid size-10 place-items-center rounded-xl bg-tile text-fg3 shadow-[var(--shadow-hi)]">
              <Library className="size-4" strokeWidth={ICON_STROKE} aria-hidden="true" />
            </span>
            <h2 className="mt-4 text-ui font-medium text-fg">{m["library.emptyTitle"]()}</h2>
            <p className="mt-1.5 max-w-[24rem] text-ui leading-6 text-fg2">
              {m["library.emptyDescription"]()}
            </p>
          </div>
        ) : visibleRows.length === 0 ? (
          <div className="flex flex-1 flex-col items-center justify-center py-16 text-center">
            <span className="grid size-10 place-items-center rounded-xl bg-tile text-fg3 shadow-[var(--shadow-hi)]">
              <SearchX className="size-4" strokeWidth={ICON_STROKE} aria-hidden="true" />
            </span>
            <h2 className="mt-4 text-ui font-medium text-fg">{m["library.noMatchesTitle"]()}</h2>
            <p className="mt-1.5 max-w-[24rem] text-ui leading-6 text-fg2">
              {m["library.noMatchesDescription"]()}
            </p>
          </div>
        ) : (
          <LibraryTable rows={visibleRows} projectId={projectId} />
        )}
      </div>
    </div>
  );
}
