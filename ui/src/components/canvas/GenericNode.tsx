import { type NodeProps } from "@xyflow/react";
import { cn } from "@/lib/utils";
import { BaseNode, type ParamRow, type SocketSlot } from "./BaseNode";
import {
  categoryTone,
  formatParamValue,
  inputSlotsFor,
  outputSlotsFor,
  paramSpecsFor,
  paramValueFor,
} from "@/lib/nodes";
import { selectNodeDef, useNodeRegistryStore } from "@/store/nodeRegistry";
import { useNodeArtifact } from "@/hooks/useNodeArtifact";
import { NodePreview } from "./NodePreview";

type GenericNodeData = {
  title?: unknown;
  tone?: unknown;
  inputs?: SocketSlot[];
  outputs?: SocketSlot[];
  parameters?: unknown[];
  params?: Record<string, unknown>;
  disabled?: unknown;
};

const PREVIEW_TYPES = new Set([
  "builtin.preview_image",
  "builtin.save_image",
  "builtin.vae_decode",
]);

/**
 * GenericNode — schema-driven React Flow node (F2-1).
 *
 * Renders ANY node type from its `NodeDefDto` metadata: header from the
 * display name, sockets from the def's input/output specs, and parameters
 * from the def's param specs (read-only chips here — editing lives in the
 * PropertiesPanel). Nodes that predate the catalog keep rendering from
 * their embedded `data` (title/tone/inputs/outputs/parameters).
 *
 * Image-producing nodes (preview_image / save_image / vae_decode) show the
 * latest run artifact inline (F5-4).
 */
export function GenericNode({ id, type, data, selected }: NodeProps) {
  const defs = useNodeRegistryStore((s) => s.defs);
  const d = (data ?? {}) as GenericNodeData;

  const def = selectNodeDef(defs, type);
  const title =
    (typeof d.title === "string" && d.title) ||
    def?.displayName ||
    type ||
    id;
  const tone =
    (typeof d.tone === "string" && d.tone) || categoryTone(def?.category);

  const inputs = inputSlotsFor(data, def);
  const outputs = outputSlotsFor(data, def);

  const specs = paramSpecsFor(data, def);
  const rows: ParamRow[] = specs.map((spec) => ({
    id: spec.id,
    label: spec.label,
    value: formatParamValue(paramValueFor(data, spec)),
  }));

  const preview = useNodeArtifact(id);
  const showPreview = Boolean(
    preview || (type != null && PREVIEW_TYPES.has(type)),
  );

  return (
    <BaseNode
      title={title}
      tone={tone}
      inputs={inputs}
      outputs={outputs}
      parameters={rows}
      selected={selected}
      disabled={d.disabled === true}
    >
      {showPreview && (
        <div
          className={cn(
            "relative overflow-hidden rounded-md border border-control-border",
            preview?.status === "ready" ? "" : "bg-surface-container-high",
          )}
        >
          <NodePreview preview={preview} />
        </div>
      )}
    </BaseNode>
  );
}
