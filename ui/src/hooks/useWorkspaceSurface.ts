import { useEffect } from "react";
import { listProjects, subscribeDocumentEvents, getBoardSnapshot, loadWorkflow } from "@/ipc";
import { workflowFromJson } from "@/lib/workflowCodec";
import { useProjectStore } from "@/store/project";
import { useBoardStore } from "@/store/board";
import { useWorkflowStore } from "@/store/workflow";

export function useWorkspaceSurface(): void {
  const activeProjectId = useProjectStore((state) => state.activeProjectId);

  useEffect(() => {
    void listProjects().then((projects) => useProjectStore.getState().hydrate(projects));
    void subscribeDocumentEvents((event) => {
      const project = useProjectStore.getState().activeProjectId;
      if (event.projectId !== project) return;
      if (event.kind === "board.changed") {
        const board = useBoardStore.getState();
        if (event.version < board.version) return;
        void getBoardSnapshot(project).then((snapshot) => {
          if (useProjectStore.getState().activeProjectId === snapshot.projectId) {
            useBoardStore.getState().hydrate(snapshot);
          }
        });
        return;
      }
      const workflow = useWorkflowStore.getState();
      if (workflow.id !== event.documentId || event.version < workflow.version) return;
      void loadWorkflow(project, workflow.id).then((json) => {
        if (useProjectStore.getState().activeProjectId !== project) return;
        const graph = workflowFromJson(json);
        useWorkflowStore.getState().hydrate(graph.nodes, graph.edges, workflow.id, graph.name, project, graph.version);
        useWorkflowStore.temporal.getState().clear();
      });
    });
  }, []);

  useEffect(() => {
    useWorkflowStore.getState().hydrate([], [], "main", "Untitled Workflow", activeProjectId, 0);
    useWorkflowStore.temporal.getState().clear();
    useBoardStore.getState().hydrate({ id: "", projectId: activeProjectId, version: 0, items: [] });
    void getBoardSnapshot(activeProjectId).then((snapshot) => {
      if (useProjectStore.getState().activeProjectId === snapshot.projectId) {
        useBoardStore.getState().hydrate(snapshot);
      }
    });
  }, [activeProjectId]);
}