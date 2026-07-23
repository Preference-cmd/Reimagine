import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  CircleDot,
  Clock3,
  Cpu,
  Loader2,
  Square,
  XCircle,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { cn } from "@/lib/utils";
import {
  useRuntimeStore,
  type RuntimeDiagnostic,
  type RuntimePhase,
} from "@/store/runtime";

const PHASE_LABEL: Record<RuntimePhase, string> = {
  idle: "Ready",
  starting: "Preparing",
  running: "Running",
  completed: "Done",
  failed: "Failed",
  cancelled: "Cancelled",
};

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

function diagnosticTone(diagnostic: RuntimeDiagnostic) {
  switch (diagnostic.severity) {
    case "error":
      return "text-status-error";
    case "warning":
      return "text-status-warning";
    default:
      return "text-on-surface-variant";
  }
}

function DiagnosticIcon({ diagnostic }: { diagnostic: RuntimeDiagnostic }) {
  if (diagnostic.severity === "info") {
    return (
      <CircleDot
        className={cn("mt-0.5 h-3.5 w-3.5", diagnosticTone(diagnostic))}
      />
    );
  }

  return (
    <AlertTriangle
      className={cn("mt-0.5 h-3.5 w-3.5", diagnosticTone(diagnostic))}
    />
  );
}

export function RuntimeIsland({
  forceCollapsed = false,
}: {
  forceCollapsed?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const phase = useRuntimeStore((s) => s.phase);
  const runId = useRuntimeStore((s) => s.runId);
  const workflowName = useRuntimeStore((s) => s.workflowName);
  const backend = useRuntimeStore((s) => s.backend);
  const device = useRuntimeStore((s) => s.device);
  const currentNode = useRuntimeStore((s) => s.currentNode);
  const progress = useRuntimeStore((s) => s.progress);
  const elapsedMs = useRuntimeStore((s) => s.elapsedMs);
  const diagnostics = useRuntimeStore((s) => s.diagnostics);
  const cancelRun = useRuntimeStore((s) => s.cancelRun);

  const Icon = phaseIcon(phase);
  const active = phase === "starting" || phase === "running";
  const expanded = open && !forceCollapsed;
  const topDiagnostic = diagnostics.find((d) => d.severity === "error") ??
    diagnostics.find((d) => d.severity === "warning") ??
    diagnostics[0];

  useEffect(() => {
    if (forceCollapsed) {
      setOpen(false);
    }
  }, [forceCollapsed]);

  const summary = useMemo(() => {
    if (phase === "idle") return `${backend} / ${device}`;
    if (phase === "completed") return `${workflowName} / ${formatElapsed(elapsedMs)}`;
    if (phase === "failed") return topDiagnostic?.message ?? "Run failed";
    return `${currentNode ?? workflowName} / ${Math.round(progress * 100)}%`;
  }, [backend, currentNode, device, elapsedMs, phase, progress, topDiagnostic, workflowName]);

  return (
    <div className="relative flex h-11 w-[min(430px,calc(100vw-32px))] justify-center">
      <div
        className={cn(
          "panel-flat absolute left-1/2 top-0 z-10 w-full min-w-0 -translate-x-1/2 overflow-hidden",
          expanded ? "rounded-[24px]" : "rounded-full",
        )}
      >
        <button
          type="button"
          aria-expanded={expanded}
          aria-label="Runtime status"
          onClick={() => {
            if (!forceCollapsed) {
              setOpen((value) => !value);
            }
          }}
          className={cn(
            "grid h-11 w-full min-w-0 cursor-pointer grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2 px-3 text-left transition-colors hover:bg-control-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30 focus-visible:ring-inset",
          )}
        >
          <span
            className={cn(
              "flex h-7 w-7 items-center justify-center rounded-full border border-outline bg-surface-container-low",
              phaseTone(phase),
            )}
          >
            <Icon
              className={cn("h-3.5 w-3.5", active && "motion-safe:animate-spin")}
            />
          </span>
          <span className="min-w-0" aria-live="polite">
            <span className="flex min-w-0 items-center gap-2">
              <span className="truncate text-body-sm font-semibold text-on-surface">
                {PHASE_LABEL[phase]}
              </span>
              {runId && (
                <span className="truncate text-caption text-on-surface-variant">
                  {runId}
                </span>
              )}
            </span>
            <span className="block truncate text-caption text-on-surface-variant">
              {summary}
            </span>
          </span>
          <ChevronDown
            className={cn(
              "h-4 w-4 shrink-0 text-on-surface-variant transition-transform",
              expanded && "rotate-180",
            )}
          />
        </button>

        {expanded && (
          <div
            aria-label="Runtime diagnostics"
            role="region"
            className="px-4 pb-3 pt-1"
          >
            {active && (
              <div className="pb-3">
                <div className="mb-2 flex items-center justify-between gap-3 text-caption">
                  <span className="truncate font-medium text-on-surface">
                    {currentNode ?? workflowName}
                  </span>
                  <button
                    type="button"
                    onClick={cancelRun}
                    className="flex h-7 shrink-0 cursor-pointer items-center gap-1.5 rounded-full border border-outline bg-surface-container-low px-2.5 font-medium text-on-surface transition-colors hover:bg-control-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30"
                  >
                    <Square className="h-3 w-3 fill-current" />
                    Cancel
                  </button>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-surface-container-high">
                  <div
                    className="h-full origin-left rounded-full bg-primary transition-transform duration-200 ease-out motion-reduce:transition-none"
                    style={{ transform: `scaleX(${Math.max(0.04, progress)})` }}
                  />
                </div>
              </div>
            )}

            <div className="grid grid-cols-3 border-t border-outline/70 py-3 text-caption">
              <RuntimeStat
                icon={Cpu}
                label="Runtime"
                value={`${backend} / ${device}`}
              />
              <RuntimeStat
                icon={Clock3}
                label="Elapsed"
                value={formatElapsed(elapsedMs)}
              />
              <RuntimeStat
                icon={CircleDot}
                label="Node"
                value={currentNode ?? "Idle"}
              />
            </div>

            <div className="border-t border-outline/70 pt-3">
              <div className="mb-2 flex items-center justify-between gap-3 text-caption">
                <span className="font-medium text-on-surface-variant">
                  Diagnostics
                </span>
                <span className="truncate text-on-surface-variant">
                  {runId ?? workflowName}
                </span>
              </div>
              {diagnostics.length > 0 ? (
                <div className="space-y-2">
                  {diagnostics.map((diagnostic) => (
                    <div
                      key={diagnostic.id}
                      className="grid grid-cols-[auto_minmax(0,1fr)] gap-2 rounded-xl bg-control-hover/60 px-2.5 py-2"
                    >
                      <DiagnosticIcon diagnostic={diagnostic} />
                      <div className="min-w-0">
                        <div className="text-caption font-medium text-on-surface">
                          {diagnostic.source}
                        </div>
                        <div className="text-caption text-on-surface-variant">
                          {diagnostic.message}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="rounded-xl bg-control-hover/60 px-2.5 py-2 text-caption text-on-surface-variant">
                  No blocking diagnostics.
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function RuntimeStat({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Cpu;
  label: string;
  value: string;
}) {
  return (
    <div className="min-w-0 pr-3 last:pr-0">
      <div className="flex items-center gap-1.5 text-on-surface-variant">
        <Icon className="h-3.5 w-3.5 shrink-0" />
        <span>{label}</span>
      </div>
      <div className="mt-1 truncate font-medium text-on-surface">{value}</div>
    </div>
  );
}
