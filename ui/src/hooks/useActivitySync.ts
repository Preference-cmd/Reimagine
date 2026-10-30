import { useEffect } from "react";
import { subscribeEvents } from "@/ipc";
import { ActivityAppendedPayloadSchema } from "@/ipc/schemas";
import { useActivityStore } from "@/store/activity";

/**
 * Keeps the activity feed for `projectId` current. Mounted once by the shell:
 * the panel may be closed, but live records must still reach the store.
 *
 * Loaded history never touches the undo/redo stacks; only appended records do.
 */
export function useActivitySync(projectId: string): void {
  useEffect(() => {
    if (!projectId) return;

    void useActivityStore.getState().load(projectId);

    const unsubscribe = subscribeEvents((event) => {
      if (event.type !== "mod" || event.kind !== "activity.appended") return;
      const payload = ActivityAppendedPayloadSchema.safeParse(event.payload);
      if (!payload.success || payload.data.projectId !== projectId) return;
      useActivityStore.getState().append(payload.data.record);
    });

    return unsubscribe;
  }, [projectId]);
}
