import { useEffect, useState } from "react";
import { Clock, FileText } from "lucide-react";
import { cn } from "@/lib/utils";
import { useWorkflowStore } from "@/store/workflow";

type RecentWorkflow = {
  id: string;
  name: string;
  lastModified: number;
};

/**
 * Recent workflows list — shows recently opened workflows.
 * Clicking one loads it into the editor.
 */
export function RecentWorkflows() {
  const [recent, setRecent] = useState<RecentWorkflow[]>([]);
  const currentId = useWorkflowStore((s) => s.id);

  useEffect(() => {
    // For now, show the current workflow as the only recent item.
    // TODO: Persist recent list to localStorage or backend.
    const current: RecentWorkflow = {
      id: currentId,
      name: useWorkflowStore.getState().name,
      lastModified: Date.now(),
    };
    setRecent([current]);
  }, [currentId]);

  if (recent.length === 0) return null;

  return (
    <div className="px-2 py-2">
      <div className="mb-1.5 flex items-center gap-1.5 px-2">
        <Clock className="h-3 w-3 text-on-surface-variant/60" />
        <span className="text-caption font-medium text-on-surface-variant/70">Recent</span>
      </div>

      <ul className="space-y-0.5">
        {recent.map((workflow) => (
          <li key={workflow.id}>
            <button
              type="button"
              className={cn(
                "flex w-full cursor-pointer items-center gap-2.5 rounded-lg px-2 py-1.5 text-left transition-colors",
                "hover:bg-control-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/20",
                workflow.id === currentId && "bg-control-hover",
              )}
            >
              <span
                className={cn(
                  "flex h-5 w-5 shrink-0 items-center justify-center rounded-md",
                  workflow.id === currentId
                    ? "bg-primary text-on-primary"
                    : "bg-control-hover text-on-surface-variant",
                )}
              >
                <FileText className="h-3 w-3" />
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-caption font-medium text-on-surface">
                  {workflow.name}
                </span>
                <span className="block truncate text-caption text-on-surface-variant/70">
                  {workflow.id}
                </span>
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
