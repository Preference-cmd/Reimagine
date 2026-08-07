import { useWorkflowStore } from "@/store/workflow";

/**
 * Minimal window title bar — macOS traffic-light controls on the left,
 * workflow name centered.  The entire bar is a native drag region so the
 * user can move the window from anywhere on the title bar.
 */
export function WindowControls() {
  const workflowName = useWorkflowStore((s) => s.name);

  return (
    <div
      data-tauri-drag-region
      className="flex h-11 shrink-0 select-none items-center border-b border-outline bg-background px-4"
      style={{ WebkitAppRegion: "drag" } as React.CSSProperties}
    >
      {/* Spacer for macOS traffic lights (rendered by the OS) */}
      <div className="w-20" />

      <span className="min-w-0 flex-1 truncate text-center text-body-sm font-medium text-on-surface-variant">
        {workflowName}
      </span>

      {/* Right spacer to balance the traffic-light area */}
      <div className="w-20" />
    </div>
  );
}
