import { toast } from "sonner";
import * as m from "$paraglide/messages";
import { toCommandError } from "@/ipc";
import {
  selectRedoEntryId,
  selectUndoEntryId,
  useActivityStore,
  type ActivityHistoryResync,
} from "@/store/activity";
import { useBoardStore } from "@/store/board";
import { useProjectStore } from "@/store/project";

export type BoardHistoryAction = "undo" | "redo";

function failureTitle(action: BoardHistoryAction): string {
  return action === "undo" ? m["activity.undoFailed"]() : m["activity.redoFailed"]();
}

function failureDescription(failure: unknown): string {
  return failure instanceof Error ? failure.message : String(failure);
}

/** Current host tops from a `conflict` error's details, if they are usable. */
function resyncFrom(
  details: Record<string, unknown> | null | undefined,
): ActivityHistoryResync | null {
  if (!details) return null;
  const undo = details.undo_entry_id;
  const redo = details.redo_entry_id;
  const isEntryId = (value: unknown): value is string | null =>
    value === null || typeof value === "string";
  if (!isEntryId(undo) || !isEntryId(redo)) return null;
  return { undoEntryId: undo, redoEntryId: redo };
}

/**
 * Run an undo or redo against the entry the activity feed believes is on top.
 *
 * The host rejects the request when that entry is not its current top, which
 * means the feed missed an event or a record failed to write. In that case
 * nothing changed: this resynchronizes the stacks to the host's tops, refetches
 * the activity page (so a lost record comes back from disk) and reloads the
 * board snapshot, then explains the outcome. Returns true only when the
 * board history actually moved.
 */
export async function runBoardHistory(action: BoardHistoryAction): Promise<boolean> {
  const projectId = useProjectStore.getState().activeProjectId;
  const activity = useActivityStore.getState();
  const expected = action === "undo" ? selectUndoEntryId(activity) : selectRedoEntryId(activity);
  const title = failureTitle(action);

  try {
    const board = useBoardStore.getState();
    const result =
      action === "undo"
        ? await board.undo(projectId, expected)
        : await board.redo(projectId, expected);
    if (!result) {
      toast.error(title);
      return false;
    }
    return true;
  } catch (failure) {
    const error = toCommandError(failure);
    const resync = error?.code === "conflict" ? resyncFrom(error.details) : null;
    if (error && resync) {
      useActivityStore.getState().resyncHistory(projectId, resync);
      try {
        await Promise.all([
          useActivityStore.getState().load(projectId, { force: true }),
          useBoardStore.getState().load(projectId),
        ]);
      } catch {
        // The notice below still explains the mismatch if a refresh read fails.
      }
      toast.error(m["activity.historyOutOfDate"](), {
        description:
          action === "undo"
            ? m["activity.historyOutOfDateUndo"]()
            : m["activity.historyOutOfDateRedo"](),
      });
      return false;
    }

    toast.error(title, {
      description: error?.message ?? failureDescription(failure),
    });
    return false;
  }
}
