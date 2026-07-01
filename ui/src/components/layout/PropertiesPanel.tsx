import { useWorkflowStore } from "@/store/workflow";
import { cn } from "@/lib/utils";
import {
  Cable,
  Hash,
  MapPin,
  SlidersHorizontal,
  X,
} from "lucide-react";

type InspectorNodeData = {
  title?: unknown;
  tone?: unknown;
  inputs?: unknown;
  outputs?: unknown;
  parameters?: unknown;
  prompt?: unknown;
};

export function PropertiesPanel() {
  const selectedNode = useWorkflowStore((s) => s.selectedNode);
  const nodes = useWorkflowStore((s) => s.nodes);
  const open = useWorkflowStore((s) => s.propertiesPanelOpen);
  const setOpen = useWorkflowStore((s) => s.setPropertiesPanelOpen);

  if (!open || !selectedNode) {
    return null;
  }

  const node = nodes.find((n) => n.id === selectedNode.id);
  const data = (node?.data ?? {}) as InspectorNodeData;
  const title = readString(data.title) ?? selectedNode.id;
  const tone = readString(data.tone) ?? "#7928ca";
  const inputs = readArray(data.inputs);
  const outputs = readArray(data.outputs);
  const parameters = readParameters(data.parameters);
  const prompt = readString(data.prompt);
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
      <div className="flex items-center justify-between border-b border-outline px-3.5 py-2.5"
      >
        <div className="flex items-center gap-2"
        >
          <SlidersHorizontal className="h-4 w-4 text-on-surface-variant" />
          <span className="text-body-md font-semibold text-on-surface"
          >
            Inspector
          </span>
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
      <div className="space-y-4 overflow-y-auto p-3.5 scrollbar-hide"
      >
        <div className="flex items-center gap-2 rounded-md border border-outline bg-surface-container-high px-2.5 py-2"
        >
          <span
            className="h-2 w-2 shrink-0 rounded-full"
            style={{ backgroundColor: tone }}
          />
          <span className="min-w-0 truncate text-body-sm text-on-surface"
          >
            {title}
          </span>
        </div>
        <div className="space-y-3"
        >
          <SectionTitle icon={Hash} label="Node metadata" />
          <div className="grid grid-cols-2 gap-y-2 text-body-sm"
          >
            {rows.map((p) => (
              <div key={p.label} className="contents"
              >
                <span className="text-on-surface-variant">{p.label}</span>
                <span className="min-w-0 truncate text-right text-on-surface"
                >
                  {p.value}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-3"
        >
          <SectionTitle icon={Cable} label="Ports" />
          <div className="grid grid-cols-2 gap-y-2 text-body-sm"
          >
            <div className="contents"
            >
              <span className="text-on-surface-variant">Inputs</span>
              <span className="text-right text-on-surface">{inputs.length}</span>
            </div>
            <div className="contents"
            >
              <span className="text-on-surface-variant">Outputs</span>
              <span className="text-right text-on-surface">{outputs.length}</span>
            </div>
          </div>
        </div>

        {(parameters.length > 0 || prompt) && (
          <div className="space-y-3"
          >
            <SectionTitle icon={MapPin} label="Values" />
            <div className="space-y-2 text-body-sm"
            >
              {prompt && (
                <div className="rounded-md border border-outline bg-surface-container-low p-2.5 text-on-surface"
                >
                  <div className="mb-1 text-on-surface-variant">Prompt</div>
                  <p className="line-clamp-4 leading-relaxed">{prompt}</p>
                </div>
              )}
              {parameters.map((p) => (
                <div
                  key={p.id}
                  className="grid grid-cols-[minmax(0,1fr)_auto] gap-3"
                >
                  <span className="min-w-0 truncate text-on-surface-variant"
                  >
                    {p.label}
                  </span>
                  <span className="max-w-32 truncate text-right text-on-surface"
                  >
                    {p.value}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function SectionTitle({
  icon: Icon,
  label,
}: {
  icon: typeof Hash;
  label: string;
}) {
  return (
    <h4 className="flex items-center gap-2 text-body-sm font-semibold text-on-surface-variant"
    >
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

function readParameters(
  value: unknown,
): Array<{ id: string; label: string; value: string }> {
  if (!Array.isArray(value)) return [];

  return value.flatMap((item, index) => {
    if (typeof item !== "object" || item == null) return [];
    const record = item as Record<string, unknown>;
    const label = readString(record.label);
    const paramValue = readString(record.value);
    if (!label || !paramValue) return [];

    return [
      {
        id: readString(record.id) ?? `${label}-${index}`,
        label,
        value: paramValue,
      },
    ];
  });
}

function formatType(type: string | null): string {
  if (!type) return "Unknown";

  return type
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/^./, (char) => char.toUpperCase());
}
