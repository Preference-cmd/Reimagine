import { NodeResizer, type NodeProps } from "@xyflow/react";
import { cn } from "@/lib/utils";
import { useUIStore } from "@/store/uiStore";
import { getCardRenderer } from "./registry";
import { MIN_ITEM_WIDTH, MIN_ITEM_HEIGHT } from "./batch";
import { boardNodeTraceClass } from "./trace";
import type { BoardItem } from "@/ipc/schemas";

export type BoardNodeData = {
  item: BoardItem;
  isEditing?: boolean;
  onStartEdit?: (id: string) => void;
  onEndEdit?: (id: string) => void;
  onUpdateNote?: (id: string, content: string) => void;
  onResizeEnd?: (id: string, size: { width: number; height: number }) => void;
};

export function BoardNode({ id, data, selected }: NodeProps) {
  const nodeData = data as unknown as BoardNodeData;
  const item = nodeData.item;
  const CardComponent = getCardRenderer(item.kind.type);
  const canvasTrace = useUIStore((state) => state.canvasTrace);

  return (
    <div
      className={cn(
        "relative h-full w-full transition-[opacity,box-shadow] duration-200 ease-ui motion-reduce:transition-none",
        boardNodeTraceClass(canvasTrace, id),
      )}
      data-tracing={canvasTrace ? String(canvasTrace.targets.includes(id)) : undefined}
      data-testid="board-node"
    >
      <NodeResizer
        minWidth={MIN_ITEM_WIDTH}
        minHeight={MIN_ITEM_HEIGHT}
        isVisible={Boolean(selected && !item.locked)}
        lineClassName="!border-accent"
        handleClassName="!h-2.5 !w-2.5 !rounded-xs !border-surface !bg-accent"
        onResizeEnd={(_event, params) => {
          nodeData.onResizeEnd?.(id, { width: params.width, height: params.height });
        }}
      />
      <CardComponent
        id={item.id}
        kind={item.kind}
        size={item.size}
        locked={item.locked}
        selected={selected}
        isEditing={nodeData.isEditing}
        onStartEdit={() => nodeData.onStartEdit?.(id)}
        onEndEdit={() => nodeData.onEndEdit?.(id)}
        onUpdateNote={(content) => nodeData.onUpdateNote?.(id, content)}
      />
    </div>
  );
}
