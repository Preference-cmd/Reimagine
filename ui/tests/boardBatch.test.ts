import { describe, test, expect } from "vitest";
import {
  createBoardBatch,
  roundPosition,
  clampSize,
  buildAddItemCommand,
  buildRemoveItemCommand,
  buildMoveItemCommand,
  buildResizeItemCommand,
  buildResizeEndCommand,
  buildUpdateNoteCommand,
  buildDragStopCommands,
  MIN_ITEM_WIDTH,
  MIN_ITEM_HEIGHT,
  DEFAULT_NOTE_SIZE,
} from "../src/board/batch";

describe("Board CommandBatch and command builders", () => {
  test("createBoardBatch constructs valid batch with default actor and provenance", () => {
    const commands = [buildRemoveItemCommand("item-1")];
    const batch = createBoardBatch(7, commands);

    expect(batch.base_version).toBe(7);
    expect(batch.actor).toEqual({ kind: "Human", id: null, label: null });
    expect(batch.provenance).toEqual({ type: "direct" });
    expect(batch.correlation_id).toBeNull();
    expect(batch.commands).toEqual(commands);
    expect(typeof batch.id).toBe("string");
    expect(batch.id.length).toBeGreaterThan(0);
    expect(Date.parse(batch.created_at)).not.toBeNaN();
  });

  test("roundPosition rounds fractional coordinates to integer values", () => {
    expect(roundPosition({ x: 12.4, y: 55.7 })).toEqual({ x: 12, y: 56 });
    expect(roundPosition({ x: -10.6, y: -0.2 })).toEqual({ x: -11, y: 0 });
  });

  test("clampSize rounds size and enforces minimum width and height", () => {
    expect(clampSize({ width: 300.2, height: 199.8 })).toEqual({ width: 300, height: 200 });
    expect(clampSize({ width: 10, height: 20 })).toEqual({
      width: MIN_ITEM_WIDTH,
      height: MIN_ITEM_HEIGHT,
    });
  });

  test("buildAddItemCommand rounds position and clamps size", () => {
    const cmd = buildAddItemCommand({
      itemId: "custom-id",
      kind: { type: "note", content: "test" },
      position: { x: 100.8, y: 200.1 },
      size: { width: 50, height: 40 },
    });

    expect(cmd).toEqual({
      type: "add_item",
      item_id: "custom-id",
      kind: { type: "note", content: "test" },
      position: { x: 101, y: 200 },
      size: { width: MIN_ITEM_WIDTH, height: MIN_ITEM_HEIGHT },
    });
  });

  test("buildAddItemCommand uses DEFAULT_NOTE_SIZE when size is omitted", () => {
    const cmd = buildAddItemCommand({
      kind: { type: "note", content: "" },
      position: { x: 0, y: 0 },
    });

    expect(cmd.size).toEqual(DEFAULT_NOTE_SIZE);
    expect(typeof cmd.item_id).toBe("string");
    expect(cmd.item_id.length).toBeGreaterThan(0);
  });

  test("buildMoveItemCommand rounds coordinates", () => {
    const cmd = buildMoveItemCommand("item-x", { x: 45.6, y: 78.2 });
    expect(cmd).toEqual({
      type: "move_item",
      item_id: "item-x",
      position: { x: 46, y: 78 },
    });
  });

  test("buildResizeItemCommand clamps dimensions", () => {
    const cmd = buildResizeItemCommand("item-y", { width: 15, height: 350.4 });
    expect(cmd).toEqual({
      type: "resize_item",
      item_id: "item-y",
      size: { width: MIN_ITEM_WIDTH, height: 350 },
    });
  });

  test("buildResizeEndCommand clamps size and skips when unchanged", () => {
    const item = { id: "item-1", size: { width: 240, height: 160 } };

    // When item is undefined -> null
    expect(buildResizeEndCommand(undefined, { width: 300, height: 200 })).toBeNull();

    // When new size rounds/clamps to current size -> null
    expect(buildResizeEndCommand(item, { width: 240.2, height: 159.8 })).toBeNull();
    expect(buildResizeEndCommand(item, { width: 240, height: 160 })).toBeNull();

    // When below minimum dimensions -> clamped to MIN_ITEM_WIDTH and MIN_ITEM_HEIGHT
    expect(buildResizeEndCommand(item, { width: 50, height: 50 })).toEqual({
      type: "resize_item",
      item_id: "item-1",
      size: { width: MIN_ITEM_WIDTH, height: MIN_ITEM_HEIGHT },
    });

    // When valid size change -> returns resize_item command
    expect(buildResizeEndCommand(item, { width: 320.4, height: 210.1 })).toEqual({
      type: "resize_item",
      item_id: "item-1",
      size: { width: 320, height: 210 },
    });
  });

  test("buildUpdateNoteCommand produces update_note command", () => {
    const cmd = buildUpdateNoteCommand("note-1", "Updated text");
    expect(cmd).toEqual({
      type: "update_note",
      item_id: "note-1",
      content: "Updated text",
    });
  });

  test("buildDragStopCommands filters unchanged nodes and rounds moved positions", () => {
    const items = [
      { id: "item-1", position: { x: 100, y: 100 } },
      { id: "item-2", position: { x: 200, y: 200 } },
      { id: "item-3", position: { x: 300, y: 300 } },
    ];

    const dragged = [
      // item-1 moved by fractional distance that rounds to same coordinates -> unchanged!
      { id: "item-1", position: { x: 100.2, y: 99.8 } },
      // item-2 moved to new position
      { id: "item-2", position: { x: 250.6, y: 280.4 } },
      // item-4 not in items -> ignored
      { id: "item-4", position: { x: 500, y: 500 } },
    ];

    const commands = buildDragStopCommands(items, dragged);
    expect(commands).toHaveLength(1);
    expect(commands[0]).toEqual({
      type: "move_item",
      item_id: "item-2",
      position: { x: 251, y: 280 },
    });
  });
});
