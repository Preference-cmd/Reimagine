import { Play, Square } from "lucide-react";
import { cn } from "@/lib/utils";
import { useRuntimeStore } from "@/store/runtime";

/**
 * Floating action button — bottom-right of the canvas.
 * Shows Play when idle, Square to cancel when running.
 * Uses offset shadow with blur for proper depth (not zero-offset halo).
 */
export function RunFab() {
  const phase = useRuntimeStore((s) => s.phase);
  const startRun = useRuntimeStore((s) => s.startRun);
  const cancelRun = useRuntimeStore((s) => s.cancelRun);

  const active = phase === "starting" || phase === "running";

  return (
    <button
      type="button"
      aria-label={active ? "Cancel run" : "Run workflow"}
      onClick={active ? cancelRun : startRun}
      className={cn(
        "absolute bottom-5 right-5 z-10 flex h-12 w-12 cursor-pointer items-center justify-center rounded-full transition-all duration-150",
        "shadow-[0_2px_8px_-2px_rgb(0_0_0/0.25),0_4px_16px_-4px_rgb(0_0_0/0.15)]",
        "hover:shadow-[0_4px_12px_-2px_rgb(0_0_0/0.3),0_8px_24px_-4px_rgb(0_0_0/0.2)]",
        "hover:scale-105 active:scale-95",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-offset-background",
        active
          ? "bg-status-error text-on-error hover:bg-status-error/90 focus-visible:ring-status-error/30"
          : "bg-primary text-on-primary hover:bg-primary/90 focus-visible:ring-primary/30",
      )}
    >
      {active ? (
        <Square className="h-5 w-5 fill-current" />
      ) : (
        <Play className="h-5 w-5 fill-current" />
      )}
    </button>
  );
}
