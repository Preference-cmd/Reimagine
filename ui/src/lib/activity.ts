import { opInvoke } from "@/ipc";
import {
  ActivityListResultSchema,
  type ActivityListResult,
  type ActivityRecord,
} from "@/ipc/schemas";

/** Default page size for `activity.list` (the host caps it at 200). */
export const ACTIVITY_PAGE_SIZE = 50;

/** History entry ids the live feed says are undoable / redoable. */
export type ActivityStacks = {
  undoStack: string[];
  redoStack: string[];
};

function pushUnique(stack: string[], entryId: string): string[] {
  return stack.includes(entryId) ? stack : [...stack, entryId];
}

/**
 * Stack transition for one live activity record.
 *
 * Only records appended while the app runs may reach the stacks: the board
 * history is in memory, so a loaded page never creates an undo target. A
 * record without a history entry id leaves the stacks untouched.
 */
export function applyActivityRecord(
  stacks: ActivityStacks,
  record: ActivityRecord,
): ActivityStacks {
  const entryId = record.historyEntryId;
  if (!entryId) return stacks;

  switch (record.kind) {
    case "applied":
      return { undoStack: pushUnique(stacks.undoStack, entryId), redoStack: [] };
    case "undone":
      return {
        undoStack: stacks.undoStack.filter((id) => id !== entryId),
        redoStack: pushUnique(stacks.redoStack, entryId),
      };
    case "redone":
      return {
        undoStack: pushUnique(stacks.undoStack, entryId),
        redoStack: stacks.redoStack.filter((id) => id !== entryId),
      };
    default:
      return stacks;
  }
}

/**
 * Appends an older page to the feed. Pages arrive newest-first, so records
 * join at the end; ids already present (a live event that raced the read) are
 * dropped.
 */
export function mergeActivityPage(
  current: ActivityRecord[],
  page: ActivityRecord[],
): ActivityRecord[] {
  if (page.length === 0) return current;

  const seen = new Set(current.map((record) => record.id));
  const merged = current.slice();
  for (const record of page) {
    if (seen.has(record.id)) continue;
    seen.add(record.id);
    merged.push(record);
  }
  return merged.length === current.length ? current : merged;
}

/** Id of the history entry the undo action targets, if any. */
export function undoEntryId(stacks: ActivityStacks): string | null {
  return stacks.undoStack[stacks.undoStack.length - 1] ?? null;
}

/** Id of the history entry the redo action targets, if any. */
export function redoEntryId(stacks: ActivityStacks): string | null {
  return stacks.redoStack[stacks.redoStack.length - 1] ?? null;
}

export function canUndo(stacks: ActivityStacks): boolean {
  return stacks.undoStack.length > 0;
}

export function canRedo(stacks: ActivityStacks): boolean {
  return stacks.redoStack.length > 0;
}

/* ───── Actor filter (all / you / agent) ───── */

export const ACTIVITY_ACTOR_FILTERS = ["all", "you", "agent"] as const;
export type ActivityActorFilter = (typeof ACTIVITY_ACTOR_FILTERS)[number];

export function isActivityActorFilter(
  value: string | null | undefined,
): value is ActivityActorFilter {
  return typeof value === "string" && (ACTIVITY_ACTOR_FILTERS as readonly string[]).includes(value);
}

/**
 * "you" is the human operator; importer/system records only appear under
 * "all", since the panel has no separate identities for them.
 */
export function activityActorMatches(
  record: Pick<ActivityRecord, "actor">,
  filter: ActivityActorFilter,
): boolean {
  if (filter === "all") return true;
  if (filter === "you") return record.actor.kind === "Human";
  return record.actor.kind === "Agent";
}

/* ───── Typed read (op_invoke + output parsing) ───── */

export type ListActivityOptions = {
  limit?: number;
  /** Id of the last record of the previous page. */
  before?: string;
};

/** One page of project activity, newest first. */
export async function listActivity(
  projectId: string,
  options: ListActivityOptions = {},
): Promise<ActivityListResult> {
  const input: { limit: number; before?: string } = {
    limit: options.limit ?? ACTIVITY_PAGE_SIZE,
  };
  if (options.before) input.before = options.before;

  const result = await opInvoke(projectId, "activity.list", input);
  return ActivityListResultSchema.parse(result.output);
}
