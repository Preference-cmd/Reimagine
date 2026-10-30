import { useEffect, useMemo, useState, type ComponentType } from "react";
import { Activity, Bot, CircleAlert, Cpu, Loader2, UserRound, X } from "lucide-react";
import { toast } from "sonner";
import { getLocale } from "$paraglide/runtime";
import * as m from "$paraglide/messages";
import { ICON_STROKE } from "@/design/tokens";
import { cn } from "@/lib/utils";
import { activityActorMatches, type ActivityActorFilter } from "@/lib/activity";
import { useActivityStore, selectRedoEntryId, selectUndoEntryId } from "@/store/activity";
import { useBoardStore } from "@/store/board";
import { useProjectStore } from "@/store/project";
import { useUIStore } from "@/store/uiStore";
import type { ActivityActor, ActivityRecord } from "@/ipc/schemas";
import { formatRelativeTime, ShellIconButton } from "./shellChrome";

type Glyph = ComponentType<{ className?: string; strokeWidth?: number; "aria-hidden"?: boolean }>;

const ACTOR_GLYPHS: Record<ActivityActor["kind"], Glyph> = {
  Human: UserRound,
  Agent: Bot,
  Importer: Cpu,
  System: Cpu,
};

function actorLabel(actor: ActivityActor): string {
  // A named agent or importer carries its own label; the kind label is only
  // the fallback.
  const label = actor.label?.trim();
  if (label) return label;

  switch (actor.kind) {
    case "Human":
      return m["activity.actorYou"]();
    case "Agent":
      return m["activity.actorAgent"]();
    case "Importer":
    case "System":
      return m["activity.actorSystem"]();
  }
}

function numericParam(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

/**
 * Localized record summary. The host stores an English fallback plus a key
 * and params; an unknown key keeps the host's sentence instead of hiding the
 * row.
 */
function activitySummary(record: ActivityRecord): string {
  const params = (record.summaryParams ?? {}) as { count?: unknown; op?: unknown };
  const count = numericParam(params.count, 1);

  switch (record.summaryKey) {
    case "activity.board.addNote":
      return count === 1
        ? m["activity.board.addNote"]()
        : m["activity.board.addNoteMany"]({ count });
    case "activity.board.addItem":
      return count === 1
        ? m["activity.board.addItem"]()
        : m["activity.board.addItemMany"]({ count });
    case "activity.board.apply":
      return count === 1 ? m["activity.board.apply"]() : m["activity.board.applyMany"]({ count });
    case "activity.board.undo":
      return m["activity.board.undo"]();
    case "activity.board.redo":
      return m["activity.board.redo"]();
    case "activity.op":
      return typeof params.op === "string" && params.op.length > 0
        ? m["activity.op"]({ op: params.op })
        : record.summary;
    default:
      return record.summary;
  }
}

const HISTORY_BUTTON =
  "cursor-pointer rounded-xs text-caption font-medium text-fg2 underline decoration-hair-strong underline-offset-2 transition-colors hover:text-fg hover:decoration-current focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:cursor-default disabled:opacity-40 disabled:hover:text-fg2";

/**
 * Floating session panel over the content area. Its workspace-activity tab
 * renders the live feed, actor filters, row-level undo/redo and the canvas
 * trace set while a row is hovered or focused.
 */
export function SessionPanel() {
  const open = useUIStore((state) => state.sessionPanelOpen);
  const setOpen = useUIStore((state) => state.setSessionPanelOpen);
  const setCanvasTrace = useUIStore((state) => state.setCanvasTrace);
  const clearCanvasTrace = useUIStore((state) => state.clearCanvasTrace);
  const projects = useProjectStore((state) => state.projects);
  const projectId = useProjectStore((state) => state.activeProjectId);

  const records = useActivityStore((state) => state.records);
  const loading = useActivityStore((state) => state.loading);
  const loadingMore = useActivityStore((state) => state.loadingMore);
  const error = useActivityStore((state) => state.error);
  const hasMore = useActivityStore((state) => state.hasMore);
  const undoEntryId = useActivityStore(selectUndoEntryId);
  const redoEntryId = useActivityStore(selectRedoEntryId);

  const [filter, setFilter] = useState<ActivityActorFilter>("all");
  const [pending, setPending] = useState<"undo" | "redo" | null>(null);

  const visibleRecords = useMemo(
    () => records.filter((record) => activityActorMatches(record, filter)),
    [records, filter],
  );

  // Only the newest visible record for the entry carries the action: an entry
  // appears again in the feed when it is redone, and the older row must not
  // offer a second undo.
  const undoRecordId = useMemo(() => {
    if (!undoEntryId) return null;
    return (
      visibleRecords.find(
        (record) =>
          (record.kind === "applied" || record.kind === "redone") &&
          record.historyEntryId === undoEntryId,
      )?.id ?? null
    );
  }, [undoEntryId, visibleRecords]);

  const redoRecordId = useMemo(() => {
    if (!redoEntryId) return null;
    return (
      visibleRecords.find(
        (record) => record.kind === "undone" && record.historyEntryId === redoEntryId,
      )?.id ?? null
    );
  }, [redoEntryId, visibleRecords]);

  useEffect(() => {
    if (!open) clearCanvasTrace();
  }, [open, clearCanvasTrace]);

  useEffect(() => () => clearCanvasTrace(), [clearCanvasTrace]);

  if (!open) return null;

  const title = projects.find((project) => project.id === projectId)?.name ?? projectId;

  function traceRecord(record: ActivityRecord) {
    if (record.targets.length === 0) return;
    const itemIds = new Set(useBoardStore.getState().items.map((item) => item.id));
    // A trace that matches nothing would dim the whole canvas; leave it be.
    if (!record.targets.some((id) => itemIds.has(id))) return;
    setCanvasTrace({ targets: record.targets, actorKind: record.actor.kind });
  }

  function retryPage() {
    const store = useActivityStore.getState();
    if (!store.loaded) {
      if (projectId) void store.load(projectId);
      return;
    }
    void store.loadMore();
  }

  async function runHistory(action: "undo" | "redo") {
    if (pending) return;
    setPending(action);
    try {
      const board = useBoardStore.getState();
      const result =
        action === "undo"
          ? await board.undo(useProjectStore.getState().activeProjectId)
          : await board.redo(useProjectStore.getState().activeProjectId);
      if (!result) {
        toast.error(action === "undo" ? m["activity.undoFailed"]() : m["activity.redoFailed"]());
      }
    } catch (failure) {
      toast.error(action === "undo" ? m["activity.undoFailed"]() : m["activity.redoFailed"](), {
        description: failure instanceof Error ? failure.message : String(failure),
      });
    } finally {
      setPending(null);
    }
  }

  const chips: Array<{ value: ActivityActorFilter; label: string }> = [
    { value: "all", label: m["activity.filterAll"]() },
    { value: "you", label: m["activity.filterYou"]() },
    { value: "agent", label: m["activity.filterAgent"]() },
  ];

  return (
    <aside
      aria-label={m["shell.sessionPanel"]()}
      data-testid="session-panel"
      className="absolute top-4 right-4 bottom-4 z-20 flex w-[var(--spacing-panel)] flex-col overflow-hidden rounded-[18px] bg-surface shadow-[var(--shadow-bezel),var(--shadow-3),var(--shadow-hi)]"
    >
      <div className="flex items-center gap-0.5 px-4 pt-3 pb-2">
        <h2 className="min-w-0 flex-1 truncate text-sm font-semibold tracking-[-0.01em] text-fg">
          {title}
        </h2>
        <ShellIconButton label={m["shell.closePanel"]()} onClick={() => setOpen(false)}>
          <X className="size-4" strokeWidth={ICON_STROKE} />
        </ShellIconButton>
      </div>

      <div className="mx-4 mb-3 flex rounded-[11px] bg-app p-[3px] shadow-[inset_0_0_0_1px_var(--color-hair-soft)]">
        <div className="flex h-7 flex-1 items-center justify-center gap-1.5 rounded-sm bg-surface text-xs font-medium text-fg shadow-[0_0_0_1px_var(--color-hair),var(--shadow-1)]">
          <Activity className="size-3.5 text-fg2" strokeWidth={ICON_STROKE} />
          {m["shell.activity"]()}
        </div>
      </div>

      <div className="flex min-h-0 flex-1 flex-col shadow-[inset_0_1px_0_var(--color-hair-soft)]">
        <div
          role="group"
          aria-label={m["activity.filterLabel"]()}
          className="flex shrink-0 items-center gap-1 px-4 py-3"
        >
          {chips.map((chip) => {
            const selected = filter === chip.value;
            return (
              <button
                key={chip.value}
                type="button"
                aria-pressed={selected}
                data-testid={`activity-chip-${chip.value}`}
                onClick={() => setFilter(chip.value)}
                className={cn(
                  "inline-flex h-7 cursor-pointer items-center rounded-full px-2.5 text-caption transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent",
                  selected
                    ? "bg-accent-tint font-medium text-accent shadow-[inset_0_0_0_1px_var(--color-accent-line)]"
                    : "text-fg2 shadow-[inset_0_0_0_1px_var(--color-hair)] hover:bg-hover hover:text-fg",
                )}
              >
                {chip.label}
              </button>
            );
          })}
        </div>

        {loading && records.length === 0 ? (
          <div
            role="status"
            data-testid="activity-loading"
            className="flex flex-1 items-center justify-center gap-2 pb-12 text-ui text-fg2"
          >
            <Loader2 className="size-4 animate-spin" strokeWidth={ICON_STROKE} aria-hidden="true" />
            {m["activity.loading"]()}
          </div>
        ) : error && records.length === 0 ? (
          <div
            role="alert"
            data-testid="activity-error"
            className="flex flex-1 flex-col items-center justify-center px-6 pb-12 text-center"
          >
            <span className="grid size-9 place-items-center rounded-lg bg-tile text-err shadow-[var(--shadow-hi)]">
              <CircleAlert className="size-4" strokeWidth={ICON_STROKE} aria-hidden="true" />
            </span>
            <p className="mt-3 text-ui font-medium text-fg">{m["activity.errorTitle"]()}</p>
            <p className="mt-1 max-w-56 text-xs leading-5 text-fg2">
              {m["activity.errorDescription"]()}
            </p>
            <p className="mt-2 max-w-full truncate font-mono text-caption text-fg2">{error}</p>
            <button
              type="button"
              data-testid="activity-retry"
              onClick={() => void useActivityStore.getState().load(projectId)}
              className="mt-4 inline-flex h-8 cursor-pointer items-center rounded-full bg-surface px-3 text-caption font-medium text-fg shadow-[inset_0_0_0_1px_var(--color-hair-strong),var(--shadow-hi)] transition-colors hover:bg-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            >
              {m["activity.retry"]()}
            </button>
          </div>
        ) : records.length === 0 ? (
          <div
            data-testid="activity-empty"
            className="flex flex-1 flex-col items-center justify-center px-6 pb-12 text-center"
          >
            <span className="grid size-9 place-items-center rounded-lg bg-tile text-fg3 shadow-[var(--shadow-hi)]">
              <Activity className="size-4" strokeWidth={ICON_STROKE} />
            </span>
            <p className="mt-3 text-ui font-medium text-fg">{m["shell.noActivity"]()}</p>
            <p className="mt-1 max-w-56 text-xs leading-5 text-fg2">
              {m["shell.noActivityHint"]()}
            </p>
          </div>
        ) : visibleRecords.length === 0 ? (
          <div
            data-testid="activity-no-matches"
            className="flex flex-1 flex-col items-center justify-center px-6 pb-12 text-center"
          >
            <span className="grid size-9 place-items-center rounded-lg bg-tile text-fg3 shadow-[var(--shadow-hi)]">
              <Activity className="size-4" strokeWidth={ICON_STROKE} />
            </span>
            <p className="mt-3 max-w-56 text-ui leading-5 text-fg2">{m["activity.noMatches"]()}</p>
          </div>
        ) : (
          <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
            <ul aria-label={m["activity.listLabel"]()} className="m-0 list-none p-0">
              {visibleRecords.map((record) => {
                const agent = record.actor.kind === "Agent";
                const Glyph = ACTOR_GLYPHS[record.actor.kind];
                const summary = activitySummary(record);
                const relative = formatRelativeTime(record.at, getLocale());
                const undo = record.id === undoRecordId;
                const redo = record.id === redoRecordId;
                // The applied row whose entry sits on the redo stack is the
                // one shown as undone; the undo record itself is an action.
                const undoneApplied =
                  record.kind === "applied" &&
                  redoEntryId !== null &&
                  record.historyEntryId === redoEntryId;

                return (
                  <li
                    key={record.id}
                    data-testid="activity-row"
                    data-activity-id={record.id}
                    tabIndex={0}
                    onPointerEnter={() => traceRecord(record)}
                    onPointerLeave={clearCanvasTrace}
                    onFocus={() => traceRecord(record)}
                    onBlur={clearCanvasTrace}
                    className="flex gap-2.5 rounded-md px-2 py-2.5 transition-colors hover:bg-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                  >
                    <span
                      aria-hidden="true"
                      className={cn(
                        "mt-0.5 grid size-6 shrink-0 place-items-center rounded-[7px] bg-tile text-fg2 shadow-[var(--shadow-hi)]",
                        agent &&
                          "bg-agent-soft text-agent shadow-[inset_0_0_0_1px_var(--color-agent-line)]",
                      )}
                    >
                      <Glyph className="size-3.5" strokeWidth={ICON_STROKE} />
                    </span>

                    <div className="min-w-0 flex-1">
                      <p
                        className={cn(
                          "text-ui leading-5 text-fg",
                          undoneApplied && "text-fg2 line-through",
                        )}
                      >
                        <b className={cn("font-medium", agent && "text-agent")}>
                          {actorLabel(record.actor)}
                        </b>{" "}
                        <span>{summary}</span>
                      </p>
                      <div className="mt-1 flex flex-wrap items-center gap-2 text-caption text-fg2">
                        {relative && <span>{relative}</span>}
                        {record.kind === "undone" && <span>{m["activity.undone"]()}</span>}
                        {undo && (
                          <button
                            type="button"
                            data-testid="activity-undo"
                            disabled={pending !== null}
                            onClick={() => void runHistory("undo")}
                            aria-label={m["activity.undoNamed"]({ summary })}
                            className={HISTORY_BUTTON}
                          >
                            {m["activity.undo"]()}
                          </button>
                        )}
                        {redo && (
                          <button
                            type="button"
                            data-testid="activity-redo"
                            disabled={pending !== null}
                            onClick={() => void runHistory("redo")}
                            aria-label={m["activity.redoNamed"]({ summary })}
                            className={HISTORY_BUTTON}
                          >
                            {m["activity.redo"]()}
                          </button>
                        )}
                      </div>
                    </div>
                  </li>
                );
              })}
            </ul>

            {error && (
              <div
                role="alert"
                data-testid="activity-page-error"
                className="mx-2 mt-2 flex items-center gap-2 rounded-sm bg-tile px-2.5 py-2 text-caption text-fg2"
              >
                <CircleAlert
                  className="size-3.5 shrink-0 text-err"
                  strokeWidth={ICON_STROKE}
                  aria-hidden="true"
                />
                <span className="min-w-0 flex-1 truncate">{error}</span>
                <button
                  type="button"
                  data-testid="activity-page-retry"
                  onClick={retryPage}
                  className="cursor-pointer rounded-xs font-medium text-fg underline decoration-hair-strong underline-offset-2 hover:decoration-current focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                >
                  {m["activity.retry"]()}
                </button>
              </div>
            )}

            {hasMore && (
              <div className="px-2 pt-2">
                <button
                  type="button"
                  data-testid="activity-load-more"
                  disabled={loadingMore}
                  onClick={() => void useActivityStore.getState().loadMore()}
                  className="inline-flex h-8 w-full cursor-pointer items-center justify-center gap-1.5 rounded-full text-caption font-medium text-fg2 shadow-[inset_0_0_0_1px_var(--color-hair)] transition-colors hover:bg-hover hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:cursor-default disabled:opacity-50"
                >
                  {loadingMore && (
                    <Loader2
                      className="size-3.5 animate-spin"
                      strokeWidth={ICON_STROKE}
                      aria-hidden="true"
                    />
                  )}
                  {loadingMore ? m["activity.loadingMore"]() : m["activity.loadMore"]()}
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </aside>
  );
}
