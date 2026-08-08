import { useWorkflowStore } from "@/store/workflow";
import { useUIStore } from "@/store/uiStore";
import { useNodeRegistryStore, selectNodeDef } from "@/store/nodeRegistry";
import { cn } from "@/lib/utils";
import { Slider } from "@/components/ui/slider";
import type { ReactNode } from "react";
import {
  coerceParamValue,
  formatParamValue,
  paramSpecsFor,
  paramValueFor,
  type ParamSpecLike,
  type ParamValue,
} from "@/lib/nodes";
import { Cable, Hash, MapPin, SlidersHorizontal, X } from "lucide-react";

type InspectorNodeData = {
  title?: unknown;
  tone?: unknown;
  inputs?: unknown;
  outputs?: unknown;
  parameters?: unknown;
  prompt?: unknown;
};

const controlClass =
  "w-full rounded-md border border-control-border bg-surface-container-high px-2.5 py-1.5 text-body-sm text-on-surface placeholder:text-on-surface-variant focus:outline-none focus:ring-1 focus:ring-control-active";

export function PropertiesPanel() {
  const selectedNode = useWorkflowStore((s) => s.selectedNode);
  const nodes = useWorkflowStore((s) => s.nodes);
  const open = useUIStore((s) => s.propertiesDrawerOpen);
  const setOpen = useUIStore((s) => s.setPropertiesDrawerOpen);
  const defs = useNodeRegistryStore((s) => s.defs);

  if (!open || !selectedNode) {
    return null;
  }

  const node = nodes.find((n) => n.id === selectedNode.id);
  const data = (node?.data ?? {}) as InspectorNodeData;
  const title = readString(data.title) ?? selectedNode.id;
  const tone = readString(data.tone) ?? "#7928ca";
  const inputs = readArray(data.inputs);
  const outputs = readArray(data.outputs);
  const prompt = readString(data.prompt);

  const def = node ? selectNodeDef(defs, node.type) : undefined;
  const specs = node ? paramSpecsFor(node, def) : [];

  const rows = [
    { label: "ID", value: selectedNode.id },
    { label: "Type", value: formatType(selectedNode.type) },
    ...(node
      ? [
          {
            label: "Position",
            value: `${Math.round(node.position.x)}, ${Math.round(node.position.y)}`,
          },
        ]
      : []),
  ];

  return (
    <div
      className={cn(
        "overlay-slot-inspector panel-raised pointer-events-auto flex max-h-[min(580px,calc(100vh-96px))] w-64 flex-col rounded-lg",
      )}
    >
      <div className="flex items-center justify-between border-b border-outline px-3.5 py-2.5">
        <div className="flex items-center gap-2">
          <SlidersHorizontal className="h-4 w-4 text-on-surface-variant" />
          <span className="text-body-md font-semibold text-on-surface">Inspector</span>
        </div>
        <button
          onClick={() => setOpen(false)}
          aria-label="Close inspector"
          className="rounded-md p-1 text-on-surface-variant hover:bg-control-hover hover:text-on-surface"
          type="button"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
      <div className="space-y-4 overflow-y-auto p-3.5 scrollbar-hide">
        <div className="flex items-center gap-2 rounded-md border border-outline bg-surface-container-high px-2.5 py-2">
          <span className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: tone }} />
          <span className="min-w-0 truncate text-body-sm text-on-surface">{title}</span>
        </div>
        <div className="space-y-3">
          <SectionTitle icon={Hash} label="Node metadata" />
          <div className="grid grid-cols-2 gap-y-2 text-body-sm">
            {rows.map((p) => (
              <div key={p.label} className="contents">
                <span className="text-on-surface-variant">{p.label}</span>
                <span className="min-w-0 truncate text-right text-on-surface">{p.value}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-3">
          <SectionTitle icon={Cable} label="Ports" />
          <div className="grid grid-cols-2 gap-y-2 text-body-sm">
            <div className="contents">
              <span className="text-on-surface-variant">Inputs</span>
              <span className="text-right text-on-surface">{inputs.length}</span>
            </div>
            <div className="contents">
              <span className="text-on-surface-variant">Outputs</span>
              <span className="text-right text-on-surface">{outputs.length}</span>
            </div>
          </div>
        </div>

        {(specs.length > 0 || prompt) && node && (
          <div className="space-y-3">
            <SectionTitle icon={MapPin} label="Values" />
            <div className="space-y-3 text-body-sm">
              {prompt && <PromptField nodeId={node.id} prompt={prompt} />}
              {specs.map((spec) => (
                <ParamField key={spec.id} nodeId={node.id} spec={spec} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ───── Prompt (legacy free-text field, F2-4) ───── */

function PromptField({ nodeId, prompt }: { nodeId: string; prompt: string }) {
  const updateNodePrompt = useWorkflowStore((s) => s.updateNodePrompt);
  return (
    <div className="space-y-1">
      <FieldLabel>Prompt</FieldLabel>
      <textarea
        className={cn(controlClass, "min-h-20 resize-y leading-relaxed")}
        value={prompt}
        placeholder="Describe the image…"
        onChange={(e) => updateNodePrompt(nodeId, e.target.value)}
      />
    </div>
  );
}

/* ───── Per-kind editable control (F2-4) ───── */

/** Read the live value for one param of one node (zustand selector). */
function useNodeParamValue(nodeId: string, spec: ParamSpecLike): ParamValue | undefined {
  return useWorkflowStore((s) => {
    const node = s.nodes.find((n) => n.id === nodeId);
    return node ? paramValueFor(node, spec) : undefined;
  });
}

function ParamField({ nodeId, spec }: { nodeId: string; spec: ParamSpecLike }) {
  const value = useNodeParamValue(nodeId, spec);
  const updateNodeParams = useWorkflowStore((s) => s.updateNodeParams);
  const commit = (raw: string | number | boolean) =>
    updateNodeParams(nodeId, { [spec.id]: coerceParamValue(raw, spec.kind) });

  const kind = spec.kind;

  switch (kind) {
    case "int":
      return (
        <div className="space-y-1">
          <FieldLabel>{spec.label}</FieldLabel>
          <input
            type="number"
            className={controlClass}
            value={String(value ?? "")}
            step={spec.step ?? 1}
            min={spec.min}
            max={spec.max}
            onChange={(e) => commit(e.target.value)}
          />
        </div>
      );
    case "float":
      return <FloatField spec={spec} value={value} onCommit={commit} />;
    case "bool":
      return (
        <div className="flex items-center justify-between gap-2">
          <FieldLabel>{spec.label}</FieldLabel>
          <SwitchControl checked={value === true} onCheckedChange={(checked) => commit(checked)} />
        </div>
      );
    case "select":
      return <SelectField spec={spec} value={value} onCommit={commit} />;
    case "string":
    case "text":
      return (
        <div className="space-y-1">
          <FieldLabel>{spec.label}</FieldLabel>
          <textarea
            className={cn(
              controlClass,
              "min-h-14 resize-y leading-relaxed",
              kind === "string" && "min-h-8",
            )}
            value={String(value ?? "")}
            placeholder={spec.label}
            onChange={(e) => commit(e.target.value)}
          />
        </div>
      );
    default:
      // image / unhandled kinds — read-only display.
      return (
        <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-3">
          <FieldLabel>{spec.label}</FieldLabel>
          <span className="max-w-32 truncate text-right text-on-surface">
            {formatParamValue(value)}
          </span>
        </div>
      );
  }
}

function FloatField({
  spec,
  value,
  onCommit,
}: {
  spec: ParamSpecLike;
  value: ParamValue | undefined;
  onCommit: (raw: number) => void;
}) {
  const numeric = typeof value === "number" ? value : parseFloat(String(value ?? ""));
  const current = Number.isFinite(numeric) ? numeric : Number(spec.default ?? 0);
  // Constraint data (min/max/step) comes from the backend DTO when the
  // node declares it; fall back to a range that always contains the
  // current value and a small default step.
  const min = spec.min ?? Math.min(0, current - 1);
  const max = spec.max ?? Math.max(1, current * 2);
  const step = spec.step ?? 0.01;

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-2">
        <FieldLabel>{spec.label}</FieldLabel>
        <span className="text-body-sm tabular-nums text-on-surface">
          {formatParamValue(current)}
        </span>
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[current]}
        onValueChange={([next]) => onCommit(next)}
        className="py-1"
      />
    </div>
  );
}

function SelectField({
  spec,
  value,
  onCommit,
}: {
  spec: ParamSpecLike;
  value: ParamValue | undefined;
  onCommit: (raw: string) => void;
}) {
  const options = spec.options ?? [];
  const current = String(value ?? spec.default ?? "");

  // No option list available (node declares no options constraint) —
  // degrade to a text field.
  if (options.length === 0) {
    return (
      <div className="space-y-1">
        <FieldLabel>{spec.label}</FieldLabel>
        <input
          type="text"
          className={controlClass}
          value={current}
          onChange={(e) => onCommit(e.target.value)}
        />
      </div>
    );
  }

  return (
    <div className="space-y-1">
      <FieldLabel>{spec.label}</FieldLabel>
      <select
        className={cn(
          controlClass,
          "appearance-none bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2210%22%20height%3D%226%22%3E%3Cpath%20d%3D%22M1%201l4%204%204-4%22%20stroke%3D%22%23666%22%20stroke-width%3D%221.5%22%20fill%3D%22none%22%2F%3E%3C%2Fsvg%3E')] bg-[position:right_0.75rem_center] bg-no-repeat pr-8",
        )}
        value={current}
        onChange={(e) => onCommit(e.target.value)}
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </div>
  );
}

function SwitchControl({
  checked,
  onCheckedChange,
}: {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onCheckedChange(!checked)}
      className={cn(
        "relative h-5 w-9 shrink-0 rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30",
        checked ? "bg-status-success" : "bg-control-border",
      )}
    >
      <span
        className={cn(
          "absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-white shadow-sm transition-transform",
          checked && "translate-x-4",
        )}
      />
    </button>
  );
}

function FieldLabel({ children }: { children: ReactNode }) {
  return <span className="text-body-sm leading-none text-on-surface-variant">{children}</span>;
}

function SectionTitle({ icon: Icon, label }: { icon: typeof Hash; label: string }) {
  return (
    <h4 className="flex items-center gap-2 text-body-sm font-semibold text-on-surface-variant">
      <Icon className="h-3.5 w-3.5 text-on-surface-variant/60" />
      {label}
    </h4>
  );
}

function readString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function readArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function formatType(type: string | null): string {
  if (!type) return "Unknown";

  return type.replace(/([a-z])([A-Z])/g, "$1 $2").replace(/^./, (char) => char.toUpperCase());
}
