import { useWorkflowStore } from "@/store/workflow";
import { MousePointer2, X } from "lucide-react";

const DEMO_PARAMETERS: Record<string, { label: string; value: string }[]> = {
  ksampler: [
    { label: "ID", value: "#KSampler" },
    { label: "Type", value: "KSampler" },
    { label: "Title", value: "KSampler" },
  ],
};

export function PropertiesPanel() {
  const selectedNode = useWorkflowStore((s) => s.selectedNode);
  const open = useWorkflowStore((s) => s.propertiesPanelOpen);
  const setOpen = useWorkflowStore((s) => s.setPropertiesPanelOpen);

  if (!open) return null;

  if (!selectedNode) {
    return (
      <div className="panel-glass absolute right-4 top-[72px] bottom-4 z-30 flex w-72 flex-col rounded-2xl shadow-2xl">
        <div className="flex items-center justify-between border-b border-white/5 p-5">
          <span className="text-body-md font-semibold text-zinc-200">
            Properties
          </span>
          <button
            onClick={() => setOpen(false)}
            className="rounded-md p-1 text-zinc-500 hover:bg-white/5 hover:text-zinc-300"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="flex flex-1 flex-col items-center justify-center gap-3 p-8 text-center">
          <MousePointer2 className="h-8 w-8 text-zinc-700" />
          <div className="space-y-1">
            <p className="text-body-sm font-medium text-zinc-400">
              No node selected
            </p>
            <p className="text-body-sm text-zinc-600">
              Click a node on the canvas to inspect its parameters.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const params = DEMO_PARAMETERS[selectedNode.type ?? ""] ?? [
    { label: "ID", value: `#${selectedNode.id}` },
    { label: "Type", value: selectedNode.type ?? "Unknown" },
  ];

  return (
    <div className="panel-glass absolute right-4 top-[72px] bottom-4 z-30 flex w-72 flex-col rounded-2xl shadow-2xl">
      <div className="flex items-center justify-between border-b border-white/5 p-5">
        <span className="text-body-md font-semibold text-zinc-200">
          Properties
        </span>
        <button
          onClick={() => setOpen(false)}
          className="rounded-md p-1 text-zinc-500 hover:bg-white/5 hover:text-zinc-300"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
      <div className="space-y-6 overflow-y-auto p-5 scrollbar-hide">
        <div className="flex items-center gap-2 rounded-xl border border-white/5 bg-zinc-900/50 p-2">
          <span className="h-2 w-2 rounded-full bg-purple-500" />
          <span className="text-body-sm text-zinc-200">
            {selectedNode.type ?? "Node"}
          </span>
        </div>
        <div className="space-y-3">
          <h4 className="text-label-caps font-bold uppercase tracking-widest text-zinc-500">
            Node
          </h4>
          <div className="grid grid-cols-2 gap-y-2 text-body-sm">
            {params.map((p) => (
              <div key={p.label} className="contents">
                <span className="text-zinc-600">{p.label}</span>
                <span className="text-right text-zinc-300">{p.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}