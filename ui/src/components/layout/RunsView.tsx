import {
  CheckCircle2,
  CircleDot,
  Clock3,
  Cpu,
  Loader2,
  XCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useRuntimeStore, type RuntimePhase } from "@/store/runtime";

function phaseTone(phase: RuntimePhase) {
  switch (phase) {
    case "starting":
    case "running":
      return "text-status-running";
    case "completed":
      return "text-status-success";
    case "failed":
      return "text-status-error";
    default:
      return "text-status-ready";
  }
}

function phaseIcon(phase: RuntimePhase) {
  switch (phase) {
    case "starting":
    case "running":
      return Loader2;
    case "completed":
      return CheckCircle2;
    case "failed":
      return XCircle;
    default:
      return CircleDot;
  }
}

function formatElapsed(ms: number) {
  if (ms <= 0) return "0s";
  const seconds = Math.round(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${String(seconds % 60).padStart(2, "0")}s`;
}

/**
 * RunsView — full-page run status & history.
 * Shows current run status, progress, and diagnostics.
 */
export function RunsView() {
  const phase = useRuntimeStore((s) => s.phase);
  const runId = useRuntimeStore((s) => s.runId);
  const workflowName = useRuntimeStore((s) => s.workflowName);
  const backend = useRuntimeStore((s) => s.backend);
  const device = useRuntimeStore((s) => s.device);
  const currentNode = useRuntimeStore((s) => s.currentNode);
  const progress = useRuntimeStore((s) => s.progress);
  const elapsedMs = useRuntimeStore((s) => s.elapsedMs);
  const diagnostics = useRuntimeStore((s) => s.diagnostics);

  const active = phase === "starting" || phase === "running";
  const Icon = phaseIcon(phase);

  return (
    <div className="flex h-full flex-col bg-background p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-display-sm font-semibold text-on-surface">
          Runs
        </h1>
        <p className="mt-1 text-body-sm text-on-surface-variant">
          Runtime history & status
        </p>
      </div>

      {/* Current run status */}
      <div className="mb-6 rounded-xl border border-outline bg-surface p-4">
        <div className="flex items-center gap-3">
          <span
            className={cn(
              "flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-outline bg-surface-container-low",
              phaseTone(phase),
            )}
          >
            <Icon
              className={cn("h-5 w-5", active && "motion-safe:animate-spin")}
            />
          </span>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span className="text-body-sm font-semibold text-on-surface">
                {phase === "idle" ? "No active run" : workflowName}
              </span>
              {runId && (
                <span className="rounded-full bg-control-hover px-2 py-0.5 text-caption text-on-surface-variant">
                  {runId}
                </span>
              )}
            </div>
            <div className="mt-0.5 text-caption text-on-surface-variant">
              {phase === "idle"
                ? `${backend} / ${device}`
                : `${currentNode ?? "..."} · ${Math.round(progress * 100)}% · ${formatElapsed(elapsedMs)}`}
            </div>
          </div>
        </div>

        {active && (
          <div className="mt-3">
            <div className="h-1.5 overflow-hidden rounded-full bg-surface-container-high">
              <div
                className="h-full origin-left rounded-full bg-primary transition-transform duration-200 ease-out"
                style={{ transform: `scaleX(${Math.max(0.04, progress)})` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Stats grid */}
      <div className="mb-6 grid grid-cols-3 gap-3">
        <StatCard icon={Cpu} label="Backend" value={backend} />
        <StatCard icon={Clock3} label="Elapsed" value={formatElapsed(elapsedMs)} />
        <StatCard icon={CircleDot} label="Node" value={currentNode ?? "Idle"} />
      </div>

      {/* Diagnostics */}
      <div className="min-h-0 flex-1">
        <h3 className="mb-3 text-body-sm font-semibold text-on-surface">
          Diagnostics
        </h3>
        {diagnostics.length === 0 ? (
          <div className="flex items-center gap-2 rounded-xl bg-control-hover/60 px-3 py-2 text-caption text-on-surface-variant">
            <CircleDot className="h-3.5 w-3.5 opacity-30" />
            <span>No diagnostics. Run a workflow to see execution logs here.</span>
          </div>
        ) : (
          <div className="space-y-2">
            {diagnostics.map((d) => (
              <div
                key={d.id}
                className="flex items-start gap-2 rounded-xl bg-control-hover/60 px-3 py-2"
              >
                <CircleDot
                  className={cn(
                    "mt-0.5 h-3.5 w-3.5 shrink-0",
                    d.severity === "error"
                      ? "text-status-error"
                      : d.severity === "warning"
                        ? "text-status-warning"
                        : "text-on-surface-variant",
                  )}
                />
                <div className="min-w-0">
                  <div className="text-caption font-medium text-on-surface">
                    {d.source}
                  </div>
                  <div className="text-caption text-on-surface-variant">
                    {d.message}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-lg border border-outline bg-surface p-3">
      <div className="flex items-center gap-1.5 text-on-surface-variant">
        <Icon className="h-3.5 w-3.5" />
        <span className="text-caption">{label}</span>
      </div>
      <div className="mt-1 truncate text-body-sm font-medium text-on-surface">
        {value}
      </div>
    </div>
  );
}
