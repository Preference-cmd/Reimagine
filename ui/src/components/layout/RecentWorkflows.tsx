import { useEffect, useState } from "react";
import { FileText } from "lucide-react";
import { cn } from "@/lib/utils";
import { useWorkflowStore } from "@/store/workflow";

type RecentWorkflow = {
  id: string;
  name: string;
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
    };
    setRecent([current]);
  }, [currentId]);

  if (recent.length === 0) return null;

  return (
    <div className="px-3 py-2">
      <div className="mb-1 px-1">
        <span className="text-label-caps text-on-surface-variant/50">Recent</span>
      </div>

      <ul className="space-y-0.5">
        {recent.map((workflow) => (
          <li key={workflow.id}>
            <button
              type="button"
              className={cn(
                "flex h-7 w-full cursor-pointer items-center gap-2 rounded-md px-2 text-left transition-colors",
                "hover:bg-control-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/20",
                workflow.id === currentId && "bg-control-hover",
              )}
            >
              <FileText
                className={cn(
                  "h-3.5 w-3.5 shrink-0",
                  workflow.id === currentId ? "text-primary" : "text-on-surface-variant/50",
                )}
              />
              <span className="min-w-0 flex-1 truncate text-body-sm text-on-surface">
                {workflow.name}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
