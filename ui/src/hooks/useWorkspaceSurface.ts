import { useEffect } from "react";
import { toast } from "sonner";
import * as m from "$paraglide/messages";
import { listProjects, subscribeEvents, getBoardSnapshot } from "@/ipc";
import { useProjectStore } from "@/store/project";
import { useBoardStore } from "@/store/board";

export function useWorkspaceSurface(): void {
  const activeProjectId = useProjectStore((state) => state.activeProjectId);

  useEffect(() => {
    void listProjects().then((projects) => useProjectStore.getState().hydrate(projects));
    const unsubscribe = subscribeEvents((event) => {
      if (event.type !== "document" || event.kind !== "board.changed") return;
      const project = useProjectStore.getState().activeProjectId;
      if (event.projectId !== project) return;
      const board = useBoardStore.getState();
      if (event.version < board.version) return;
      void getBoardSnapshot(project)
        .then((snapshot) => {
          useBoardStore.getState().hydrateIfNewer(snapshot);
        })
        .catch(() => {
          toast.error(m["board.loadFailed"]());
        });
    });
    return () => unsubscribe();
  }, []);

  useEffect(() => {
    useBoardStore.getState().hydrate({ id: "", projectId: activeProjectId, version: 0, items: [] });
    void getBoardSnapshot(activeProjectId)
      .then((snapshot) => {
        useBoardStore.getState().hydrateIfNewer(snapshot);
      })
      .catch(() => {
        toast.error(m["board.loadFailed"]());
      });
  }, [activeProjectId]);
}
