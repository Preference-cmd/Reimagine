import { useEffect } from "react";
import { FileText } from "lucide-react";
import { cn } from "@/lib/utils";
import { useWorkflowStore } from "@/store/workflow";
import { useProjectStore } from "@/store/project";
import { useRecentWorkflowsStore } from "@/store/recentWorkflows";
import { useWorkflows } from "@/hooks/queries";
import { loadWorkflow } from "@/ipc";
import { workflowFromJson } from "@/lib/workflowCodec";
import * as m from "$paraglide/messages";

type RecentWorkflowsProps = {
  collapsed?: boolean;
};

/**
 * Recent workflows list — shows recently opened workflows.
 * Clicking one loads it into the editor.
 */
export function RecentWorkflows({ collapsed }: RecentWorkflowsProps) {
  const entries = useRecentWorkflowsStore((s) => s.entries);
  const currentId = useWorkflowStore((s) => s.id);
  const activeProjectId = useProjectStore((s) => s.activeProjectId);
  const projectEntries = entries.filter((entry) => entry.projectId === activeProjectId);
  const { data: savedWorkflows } = useWorkflows(activeProjectId);
  const updateNames = useRecentWorkflowsStore((s) => s.updateNames);

  // Validate and update names on mount
  useEffect(() => {
    if (!savedWorkflows || projectEntries.length === 0) return;

    const savedIds = new Set(savedWorkflows.map((w) => w.id));
    const validEntries = projectEntries.filter((e) => savedIds.has(e.id));

    // Remove entries for deleted workflows
    if (validEntries.length !== projectEntries.length) {
      useRecentWorkflowsStore.setState((state) => ({
        entries: [...state.entries.filter((entry) => entry.projectId !== activeProjectId), ...validEntries],
      }));
    }

    // Fetch names for entries missing them
    const needsNames = validEntries.filter((e) => !e.name || e.name === "Untitled");
    if (needsNames.length > 0) {
      Promise.all(
        needsNames.map(async (entry) => {
          try {
            const json = await loadWorkflow(activeProjectId, entry.id);
            const { name } = workflowFromJson(json);
            return { id: entry.id, name };
          } catch {
            return { id: entry.id, name: entry.id };
          }
        }),
      ).then((updates) => updateNames(activeProjectId, updates));
    }
  }, [activeProjectId, savedWorkflows, entries, updateNames]);

  const handleOpenWorkflow = async (id: string) => {
    try {
      const json = await loadWorkflow(activeProjectId, id);
      const { nodes, edges, name, version } = workflowFromJson(json);
      useWorkflowStore.getState().hydrate(nodes, edges, id, name, activeProjectId, version);
      useRecentWorkflowsStore.getState().addRecent(activeProjectId, id, name);
      useWorkflowStore.temporal.getState().clear();
    } catch (error) {
      console.error("Failed to load workflow:", error);
    }
  };

  if (projectEntries.length === 0) {
    if (collapsed) return null;
    return (
      <div className="py-2">
        <div className="mb-1 px-1">
          <span className="text-caption text-sidebar-section-header">
            {m["sidebar.recentWorkflows"]()}
          </span>
        </div>
        <p className="px-1 text-caption text-sidebar-text-muted">{m["sidebar.noRecent"]()}</p>
      </div>
    );
  }

  return (
    <div className="py-1">
      {!collapsed && (
        <div className="mb-1 px-1">
          <span className="text-caption text-sidebar-section-header">
            {m["sidebar.recentWorkflows"]()}
          </span>
        </div>
      )}

      <ul className="space-y-0.5">
        {projectEntries.slice(0, collapsed ? 3 : 10).map((entry) => (
          <li key={entry.id}>
            <button
              type="button"
              onClick={() => handleOpenWorkflow(entry.id)}
              aria-label={collapsed ? entry.name : undefined}
              title={collapsed ? entry.name : undefined}
              className={cn(
                "flex h-8 w-full cursor-pointer items-center gap-2.5 rounded-md px-2.5 text-left text-sm transition-colors duration-150",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/20",
                collapsed && "justify-center px-0",
                entry.id === currentId
                  ? "bg-sidebar-item-active text-sidebar-text-primary"
                  : "text-sidebar-text-secondary hover:bg-sidebar-item-hover hover:text-sidebar-text-primary",
              )}
            >
              <FileText
                className={cn(
                  "h-4 w-4 shrink-0",
                  entry.id === currentId
                    ? "text-sidebar-text-secondary"
                    : "text-sidebar-text-muted",
                )}
              />
              {!collapsed && <span className="min-w-0 flex-1 truncate">{entry.name}</span>}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
