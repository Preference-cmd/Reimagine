import type { BoardItemKind } from "@/ipc/schemas";

export const MIN_ITEM_WIDTH = 120;
export const MIN_ITEM_HEIGHT = 80;
export const DEFAULT_NOTE_SIZE = { width: 240, height: 160 };

export type BoardCommand =
  | {
      type: "add_item";
      item_id: string;
      kind: BoardItemKind;
      position: { x: number; y: number };
      size: { width: number; height: number };
    }
  | {
      type: "remove_item";
      item_id: string;
    }
  | {
      type: "move_item";
      item_id: string;
      position: { x: number; y: number };
    }
  | {
      type: "resize_item";
      item_id: string;
      size: { width: number; height: number };
    }
  | {
      type: "set_z";
      item_id: string;
      z: number;
    }
  | {
      type: "update_note";
      item_id: string;
      content: string;
    }
  | {
      type: "lock_item";
      item_id: string;
      locked: boolean;
    };

export type CommandActor = {
  kind: "Human" | "Agent" | "Importer" | "System";
  id: string | null;
  label: string | null;
};

export type CommandProvenance =
  | { type: "direct" }
  | { type: "agent_proposal"; proposal_id: string; approved_by: CommandActor | null }
  | { type: "import"; format: string; source: string | null }
  | { type: "migration"; from_schema_version: string };

export type CommandBatch<C = BoardCommand> = {
  id: string;
  actor: CommandActor;
  provenance: CommandProvenance;
  correlation_id: string | null;
  base_version: number;
  created_at: string;
  commands: C[];
};

export function roundPosition(position: { x: number; y: number }): { x: number; y: number } {
  return {
    x: Math.round(position.x) || 0,
    y: Math.round(position.y) || 0,
  };
}

export function clampSize(size: { width: number; height: number }): {
  width: number;
  height: number;
} {
  return {
    width: Math.max(MIN_ITEM_WIDTH, Math.round(size.width) || 0),
    height: Math.max(MIN_ITEM_HEIGHT, Math.round(size.height) || 0),
  };
}

export function createBoardBatch(
  baseVersion: number,
  commands: BoardCommand[],
  options?: {
    id?: string;
    actor?: CommandActor;
    provenance?: CommandProvenance;
    correlationId?: string | null;
    createdAt?: string;
  },
): CommandBatch<BoardCommand> {
  return {
    id: options?.id ?? crypto.randomUUID(),
    actor: options?.actor ?? { kind: "Human", id: null, label: null },
    provenance: options?.provenance ?? { type: "direct" },
    correlation_id: options?.correlationId ?? null,
    base_version: baseVersion,
    created_at: options?.createdAt ?? new Date().toISOString(),
    commands,
  };
}

export function buildAddItemCommand(params: {
  itemId?: string;
  kind: BoardItemKind;
  position: { x: number; y: number };
  size?: { width: number; height: number };
}): BoardCommand {
  return {
    type: "add_item",
    item_id: params.itemId ?? crypto.randomUUID(),
    kind: params.kind,
    position: roundPosition(params.position),
    size: clampSize(params.size ?? DEFAULT_NOTE_SIZE),
  };
}

export function buildRemoveItemCommand(itemId: string): BoardCommand {
  return {
    type: "remove_item",
    item_id: itemId,
  };
}

export function buildMoveItemCommand(
  itemId: string,
  position: { x: number; y: number },
): BoardCommand {
  return {
    type: "move_item",
    item_id: itemId,
    position: roundPosition(position),
  };
}

export function buildResizeItemCommand(
  itemId: string,
  size: { width: number; height: number },
): BoardCommand {
  return {
    type: "resize_item",
    item_id: itemId,
    size: clampSize(size),
  };
}

export function buildResizeEndCommand(
  item: { id: string; size: { width: number; height: number } } | undefined,
  newSize: { width: number; height: number },
): BoardCommand | null {
  if (!item) return null;
  const clamped = clampSize(newSize);
  if (clamped.width === item.size.width && clamped.height === item.size.height) {
    return null;
  }
  return buildResizeItemCommand(item.id, clamped);
}

export function buildUpdateNoteCommand(itemId: string, content: string): BoardCommand {
  return {
    type: "update_note",
    item_id: itemId,
    content,
  };
}

export function buildDragStopCommands(
  items: Array<{ id: string; position: { x: number; y: number } }>,
  draggedNodes: Array<{ id: string; position: { x: number; y: number } }>,
): BoardCommand[] {
  const itemMap = new Map(items.map((i) => [i.id, i.position]));
  const commands: BoardCommand[] = [];

  for (const node of draggedNodes) {
    const orig = itemMap.get(node.id);
    if (!orig) continue;
    const rounded = roundPosition(node.position);
    if (rounded.x !== orig.x || rounded.y !== orig.y) {
      commands.push(buildMoveItemCommand(node.id, rounded));
    }
  }

  return commands;
}
