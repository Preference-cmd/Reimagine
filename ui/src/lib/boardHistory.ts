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
 * In-flight host requests, keyed by project and action. A panel click and a
 * shortcut, or a double click, must not send a second request with the same
 * expectation while the first is still running.
 */
const inFlight = new Map<string, Promise<boolean>>();

async function requestHistory(
  action: BoardHistoryAction,
  projectId: string,
  expected: string,
): Promise<boolean> {
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
      // Remember the host's tops for this project even when the user has moved
      // on; the stacks return with the project's feed.
      useActivityStore.getState().resyncHistory(projectId, resync);
      if (useProjectStore.getState().activeProjectId !== projectId) {
        return false;
      }
      // One failed read must not skip the other; the stacks already follow the
      // host's tops.
      await Promise.allSettled([
        useActivityStore.getState().load(projectId, { force: true }),
        useBoardStore.getState().load(projectId),
      ]);
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

/**
 * Run an undo or redo against the entry the activity feed believes is on top.
 *
 * The activity feed is the only source of the expected id: when it belongs to
 * another project or names no top for the action, no request is sent. The
 * host's `None` behavior remains for non-UI callers, but the UI always sends
 * an expected id. Concurrent calls for the same project and action share one
 * in-flight request.
 *
 * The host rejects the request when that entry is not its current top, which
 * means the feed missed an event or a record failed to write. In that case
 * nothing changed: this resynchronizes the stacks to the host's tops and, while
 * the project is still active, refetches the activity page (so a lost record
 * comes back from disk) and reloads the board snapshot before explaining the
 * outcome. After a workspace switch the refresh and the notice are skipped, and
 * the remembered stacks wait for the project's return. Returns true only when
 * the board history actually moved.
 */
export function runBoardHistory(action: BoardHistoryAction): Promise<boolean> {
  const projectId = useProjectStore.getState().activeProjectId;
  const activity = useActivityStore.getState();
  if (activity.projectId !== projectId) return Promise.resolve(false);

  const key = `${projectId}:${action}`;
  const pending = inFlight.get(key);
  if (pending) return pending;

  const expected = action === "undo" ? selectUndoEntryId(activity) : selectRedoEntryId(activity);
  if (!expected) return Promise.resolve(false);

  const request = requestHistory(action, projectId, expected).finally(() => {
    inFlight.delete(key);
  });
  inFlight.set(key, request);
  return request;
}
