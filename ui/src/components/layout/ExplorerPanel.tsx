import { useEffect, useMemo, useState } from "react";
import {
  Boxes,
  Cable,
  CircleDot,
  Clock3,
  FileImage,
  FolderOpen,
  History,
  Image,
  MessageSquareText,
  Plus,
  Search,
  SlidersHorizontal,
  Workflow,
  X,
  type LucideIcon,
} from "lucide-react";
import type { Node } from "@xyflow/react";
import { cn } from "@/lib/utils";
import { listModels } from "@/ipc";
import type { ModelInfo } from "@/ipc";
import { useRuntimeStore } from "@/store/runtime";
import { useWorkflowStore } from "@/store/workflow";

type ExplorerView = "Graph" | "Models" | "Runs" | "Assets";

type ExplorerPanelProps = {
  open: boolean;
  view: string | null;
  onClose: () => void;
  className?: string;
};

type ExplorerRow = {
  id: string;
  label: string;
  meta?: string;
  detail?: string;
  icon: LucideIcon;
  tone?: string;
  muted?: boolean;
  active?: boolean;
  onClick?: () => void;
};

const VIEW_META: Record<
  ExplorerView,
  { title: string; eyebrow: string; icon: LucideIcon; search: string }
> = {
  Graph: {
    title: "Graph",
    eyebrow: "Current workflow",
    icon: Workflow,
    search: "Find nodes",
  },
  Models: {
    title: "Models",
    eyebrow: "Local index",
    icon: Boxes,
    search: "Find models",
  },
  Runs: {
    title: "Runs",
    eyebrow: "Runtime history",
    icon: History,
    search: "Find runs",
  },
  Assets: {
    title: "Assets",
    eyebrow: "Project files",
    icon: Image,
    search: "Find assets",
  },
};

const NODE_ICON: Record<string, LucideIcon> = {
  imageGenerator: SlidersHorizontal,
  imageOutput: FileImage,
  model: Boxes,
  prompt: MessageSquareText,
};

export function ExplorerPanel({
  open,
  view,
  onClose,
  className,
}: ExplorerPanelProps) {
  const currentView = isExplorerView(view) ? view : "Graph";
  const meta = VIEW_META[currentView];
  const HeaderIcon = meta.icon;

  const nodes = useWorkflowStore((s) => s.nodes);
  const edges = useWorkflowStore((s) => s.edges);
  const selectedNode = useWorkflowStore((s) => s.selectedNode);
  const onNodeSelect = useWorkflowStore((s) => s.onNodeSelect);

  const runtimePhase = useRuntimeStore((s) => s.phase);
  const runtimeRunId = useRuntimeStore((s) => s.runId);
  const runtimeWorkflowName = useRuntimeStore((s) => s.workflowName);
  const runtimeBackend = useRuntimeStore((s) => s.backend);
  const runtimeDevice = useRuntimeStore((s) => s.device);
  const runtimeCurrentNode = useRuntimeStore((s) => s.currentNode);
  const runtimeProgress = useRuntimeStore((s) => s.progress);
  const runtimeElapsedMs = useRuntimeStore((s) => s.elapsedMs);
  const runtimeDiagnostics = useRuntimeStore((s) => s.diagnostics);

  const [query, setQuery] = useState("");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);

  useEffect(() => {
    setQuery("");
  }, [currentView]);

  useEffect(() => {
    if (!open || currentView !== "Models") return;

    let cancelled = false;
    setModelsLoading(true);
    listModels()
      .then((result) => {
        if (!cancelled) setModels(result);
      })
      .catch(() => {
        if (!cancelled) setModels([]);
      })
      .finally(() => {
        if (!cancelled) setModelsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [currentView, open]);

  const rows = useMemo(() => {
    if (currentView === "Graph") {
      return graphRows(nodes, edges, selectedNode?.id ?? null, onNodeSelect);
    }
    if (currentView === "Models") {
      return modelRows(nodes, models, modelsLoading);
    }
    if (currentView === "Runs") {
      return runRows({
        phase: runtimePhase,
        runId: runtimeRunId,
        workflowName: runtimeWorkflowName,
        backend: runtimeBackend,
        device: runtimeDevice,
        currentNode: runtimeCurrentNode,
        progress: runtimeProgress,
        elapsedMs: runtimeElapsedMs,
        diagnostics: runtimeDiagnostics,
      });
    }
    return assetRows();
  }, [
    currentView,
    edges,
    models,
    modelsLoading,
    nodes,
    onNodeSelect,
    runtimeBackend,
    runtimeCurrentNode,
    runtimeDevice,
    runtimeDiagnostics,
    runtimeElapsedMs,
    runtimePhase,
    runtimeProgress,
    runtimeRunId,
    runtimeWorkflowName,
    selectedNode?.id,
  ]);

  const filteredRows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return rows;
    return rows.filter((row) =>
      [row.label, row.meta, row.detail]
        .filter(Boolean)
        .some((value) => value!.toLowerCase().includes(needle)),
    );
  }, [query, rows]);

  if (!open) {
    return null;
  }

  return (
    <div
      className={cn(
        "panel-raised pointer-events-auto flex max-h-[min(560px,calc(100vh-112px))] w-[248px] flex-col overflow-hidden rounded-xl",
        className,
      )}
    >
      <div className="flex items-start justify-between gap-3 px-3 py-3">
        <div className="flex min-w-0 items-center gap-2.5">
          <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-control-hover text-on-surface">
            <HeaderIcon className="h-3.5 w-3.5" />
          </span>
          <div className="min-w-0">
            <div className="truncate text-body-sm font-semibold text-on-surface">
              {meta.title}
            </div>
            <div className="truncate text-caption text-on-surface-variant">
              {meta.eyebrow}
            </div>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-0.5">
          <IconButton ariaLabel={`Add ${meta.title.toLowerCase()} item`} icon={Plus} />
          <IconButton ariaLabel="Hide explorer" icon={X} onClick={onClose} />
        </div>
      </div>

      <div className="px-3 pb-2">
        <label className="relative block">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-on-surface-variant" />
          <input
            className="h-8 w-full rounded-full border border-outline bg-surface-container-low px-3 pl-8 text-caption text-on-surface outline-none transition-[border-color,box-shadow] placeholder:text-on-surface-variant focus:border-primary/30 focus:ring-2 focus:ring-primary/10"
            placeholder={meta.search}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            type="text"
          />
        </label>
      </div>

      <div className="scrollbar-hide flex-1 overflow-y-auto px-2 pb-2">
        <ExplorerSummary view={currentView} nodes={nodes} edgesCount={edges.length} rowsCount={rows.length} />

        <ul className="mt-2 space-y-0.5">
          {filteredRows.map((row) => (
            <ExplorerListRow key={row.id} row={row} />
          ))}
        </ul>

        {filteredRows.length === 0 && (
          <div className="mx-1 mt-2 rounded-lg bg-control-hover/60 px-3 py-2 text-caption text-on-surface-variant">
            No matches.
          </div>
        )}
      </div>
    </div>
  );
}

function ExplorerSummary({
  view,
  nodes,
  edgesCount,
  rowsCount,
}: {
  view: ExplorerView;
  nodes: Node[];
  edgesCount: number;
  rowsCount: number;
}) {
  if (view === "Graph") {
    return (
      <div className="grid grid-cols-3 gap-1 px-1">
        <SummaryCell label="Nodes" value={String(nodes.length)} />
        <SummaryCell label="Edges" value={String(edgesCount)} />
        <SummaryCell label="Types" value={String(new Set(nodes.map((node) => node.type)).size)} />
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-1 px-1">
      <SummaryCell label="Shown" value={String(rowsCount)} />
      <SummaryCell label="Scope" value={view === "Models" ? "Local" : "Project"} />
    </div>
  );
}

function SummaryCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded-lg bg-control-hover/60 px-2 py-1.5">
      <div className="truncate text-caption text-on-surface-variant">{label}</div>
      <div className="truncate text-caption font-semibold text-on-surface">{value}</div>
    </div>
  );
}

function ExplorerListRow({ row }: { row: ExplorerRow }) {
  const Icon = row.icon;
  const content = (
    <>
      <span
        className={cn(
          "mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-control-hover text-on-surface-variant",
          row.active && "bg-primary text-on-primary",
        )}
      >
        <Icon className="h-3.5 w-3.5" />
      </span>
      <span className="min-w-0 flex-1">
        <span
          className={cn(
            "flex min-w-0 items-center gap-1.5 text-caption font-medium",
            row.muted ? "text-on-surface-variant/70" : "text-on-surface",
          )}
        >
          {row.tone && (
            <span
              className="h-1.5 w-1.5 shrink-0 rounded-full"
              style={{ backgroundColor: row.tone }}
            />
          )}
          <span className="truncate">{row.label}</span>
        </span>
        {(row.meta || row.detail) && (
          <span className="mt-0.5 flex min-w-0 items-center gap-1 text-caption text-on-surface-variant">
            {row.meta && <span className="truncate">{row.meta}</span>}
            {row.meta && row.detail && <span>/</span>}
            {row.detail && <span className="truncate">{row.detail}</span>}
          </span>
        )}
      </span>
    </>
  );

  if (row.onClick) {
    return (
      <li>
        <button
          type="button"
          onClick={row.onClick}
          className={cn(
            "grid w-full cursor-pointer grid-cols-[auto_minmax(0,1fr)] gap-2 rounded-lg px-2 py-1.5 text-left transition-colors hover:bg-control-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/20",
            row.active && "bg-control-hover",
          )}
        >
          {content}
        </button>
      </li>
    );
  }

  return (
    <li
      className={cn(
        "grid grid-cols-[auto_minmax(0,1fr)] gap-2 rounded-lg px-2 py-1.5",
        row.active && "bg-control-hover",
      )}
    >
      {content}
    </li>
  );
}

function IconButton({
  ariaLabel,
  icon: Icon,
  onClick,
}: {
  ariaLabel: string;
  icon: LucideIcon;
  onClick?: () => void;
}) {
  return (
    <button
      type="button"
      aria-label={ariaLabel}
      onClick={onClick}
      className="flex h-7 w-7 cursor-pointer items-center justify-center rounded-full text-on-surface-variant transition-colors hover:bg-control-hover hover:text-on-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/20"
    >
      <Icon className="h-3.5 w-3.5" />
    </button>
  );
}

function graphRows(
  nodes: Node[],
  edges: unknown[],
  selectedNodeId: string | null,
  onNodeSelect: (selection: { id: string; type: string | null } | null) => void,
): ExplorerRow[] {
  const typeRows: ExplorerRow[] = [
    {
      id: "connections",
      label: "Connections",
      meta: `${edges.length} edges`,
      icon: Cable,
      muted: edges.length === 0,
    },
    ...nodes.map((node) => {
      const title = readNodeTitle(node);
      const tone = readNodeTone(node);
      return {
        id: `node-${node.id}`,
        label: title,
        meta: formatType(node.type),
        detail: node.id,
        icon: NODE_ICON[node.type ?? ""] ?? CircleDot,
        tone,
        active: selectedNodeId === node.id,
        onClick: () => onNodeSelect({ id: node.id, type: node.type ?? null }),
      };
    }),
  ];

  return typeRows;
}

function modelRows(
  nodes: Node[],
  models: ModelInfo[],
  loading: boolean,
): ExplorerRow[] {
  const graphModels = nodes
    .filter((node) => node.type === "model")
    .map((node) => ({
      id: `used-${node.id}`,
      label: readNodeParameter(node) ?? readNodeTitle(node),
      meta: "Used in graph",
      detail: node.id,
      icon: Boxes,
      tone: readNodeTone(node),
    }));

  if (loading) {
    return [
      ...graphModels,
      {
        id: "models-loading",
        label: "Indexing models",
        meta: "Loading",
        icon: Clock3,
        muted: true,
      },
    ];
  }

  return [
    ...graphModels,
    ...models.map((model) => ({
      id: `model-${model.id}`,
      label: model.name,
      meta: model.family,
      detail: model.size,
      icon: Boxes,
      muted: graphModels.some((row) => row.label.includes(model.id)),
    })),
  ];
}

function runRows(runtime: {
  phase: string;
  runId: string | null;
  workflowName: string;
  backend: string;
  device: string;
  currentNode: string | null;
  progress: number;
  elapsedMs: number;
  diagnostics: Array<{ id: string; severity: string; message: string; source: string }>;
}): ExplorerRow[] {
  return [
    {
      id: "runtime-current",
      label: formatPhase(runtime.phase),
      meta: `${runtime.backend} / ${runtime.device}`,
      detail: runtime.runId ?? runtime.workflowName,
      icon: CircleDot,
      active: runtime.phase === "starting" || runtime.phase === "running",
    },
    {
      id: "runtime-node",
      label: runtime.currentNode ?? "No active node",
      meta: `${Math.round(runtime.progress * 100)}%`,
      detail: formatElapsed(runtime.elapsedMs),
      icon: Workflow,
      muted: runtime.phase === "idle",
    },
    ...runtime.diagnostics.map((diagnostic) => ({
      id: `diagnostic-${diagnostic.id}`,
      label: diagnostic.source,
      meta: diagnostic.severity,
      detail: diagnostic.message,
      icon: CircleDot,
      muted: diagnostic.severity === "info",
    })),
    {
      id: "history-empty",
      label: "Run history",
      meta: "Not persisted yet",
      icon: History,
      muted: true,
    },
  ];
}

function assetRows(): ExplorerRow[] {
  return [
    {
      id: "output-preview",
      label: "Generated preview",
      meta: "Image output",
      detail: "Canvas sample",
      icon: FileImage,
    },
    {
      id: "project-assets",
      label: "Project assets",
      meta: "Not indexed yet",
      icon: FolderOpen,
      muted: true,
    },
    {
      id: "imports",
      label: "Imports",
      meta: "Empty",
      icon: Image,
      muted: true,
    },
  ];
}

function isExplorerView(value: string | null): value is ExplorerView {
  return value === "Graph" || value === "Models" || value === "Runs" || value === "Assets";
}

function readNodeTitle(node: Node) {
  const title = (node.data as { title?: unknown }).title;
  return typeof title === "string" && title.trim() ? title : node.id;
}

function readNodeTone(node: Node) {
  const tone = (node.data as { tone?: unknown }).tone;
  return typeof tone === "string" ? tone : undefined;
}

function readNodeParameter(node: Node) {
  const parameters = (node.data as { parameters?: unknown }).parameters;
  if (!Array.isArray(parameters)) return null;
  const first = parameters[0] as { value?: unknown } | undefined;
  return typeof first?.value === "string" ? first.value : null;
}

function formatType(type: string | null | undefined) {
  if (!type) return "Unknown";
  return type
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/^./, (char) => char.toUpperCase());
}

function formatPhase(phase: string) {
  return phase.replace(/^./, (char) => char.toUpperCase());
}

function formatElapsed(ms: number) {
  if (ms <= 0) return "0s";
  const seconds = Math.round(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${String(seconds % 60).padStart(2, "0")}s`;
}
