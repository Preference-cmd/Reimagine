import { NodeResizer, type NodeProps } from "@xyflow/react";
import { getCardRenderer } from "./registry";
import { MIN_ITEM_WIDTH, MIN_ITEM_HEIGHT } from "./batch";
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

  return (
    <div className="relative h-full w-full">
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
