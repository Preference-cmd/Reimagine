import { describe, test, expect, beforeEach } from "vitest";
import {
  mockApplyBoardCommands,
  mockGetBoardSnapshot,
  mockOpInvoke,
  mockUndoBoard,
  mockRedoBoard,
  mockResetBoard,
} from "../src/ipc/mock";
import { subscribeEvents } from "../src/ipc";
import type { Event } from "../src/ipc/schemas";

describe("In-memory board mock", () => {
  const projectId = "test-mock-project";

  beforeEach(() => {
    mockResetBoard();
  });

  test("add_item, move_item, resize_item, set_z, update_note, lock_item, remove_item", async () => {
    const events: Event[] = [];
    const unsubscribe = subscribeEvents((e) => events.push(e));

    // Initial snapshot version is 0
    const snap0 = await mockGetBoardSnapshot({ projectId });
    expect(snap0.version).toBe(0);
    expect(snap0.items).toHaveLength(0);

    // 1. add_item
    const addResult = await mockApplyBoardCommands({
      projectId,
      commandBatch: {
        base_version: 0,
        commands: [
          {
            type: "add_item",
            item_id: "note-1",
            kind: { type: "note", content: "Initial text" },
            position: { x: 10, y: 20 },
            size: { width: 240, height: 160 },
          },
        ],
      },
    });

    expect(addResult.status).toBe("applied");
    expect(addResult.version).toBe(1);
    const snap1 = await mockGetBoardSnapshot({ projectId });
    expect(snap1.items).toHaveLength(1);
    expect(snap1.items[0].id).toBe("note-1");
    expect((snap1.items[0].kind as { content: string }).content).toBe("Initial text");

    // 2. move_item and resize_item in one batch
    const moveResizeResult = await mockApplyBoardCommands({
      projectId,
      commandBatch: {
        base_version: 1,
        commands: [
          { type: "move_item", item_id: "note-1", position: { x: 50, y: 60 } },
          { type: "resize_item", item_id: "note-1", size: { width: 300, height: 200 } },
        ],
      },
    });

    expect(moveResizeResult.status).toBe("applied");
    expect(moveResizeResult.version).toBe(2);
    const snap2 = await mockGetBoardSnapshot({ projectId });
    expect(snap2.items[0].position).toEqual({ x: 50, y: 60 });
    expect(snap2.items[0].size).toEqual({ width: 300, height: 200 });

    // 3. update_note and set_z
    const updateResult = await mockApplyBoardCommands({
      projectId,
      commandBatch: {
        base_version: 2,
        commands: [
          { type: "update_note", item_id: "note-1", content: "Modified text" },
          { type: "set_z", item_id: "note-1", z: 5 },
        ],
      },
    });
    expect(updateResult.status).toBe("applied");
    expect(updateResult.version).toBe(3);
    const snap3 = await mockGetBoardSnapshot({ projectId });
    expect((snap3.items[0].kind as { content: string }).content).toBe("Modified text");
    expect(snap3.items[0].z).toBe(5);

    // 4. lock_item
    await mockApplyBoardCommands({
      projectId,
      commandBatch: {
        base_version: 3,
        commands: [{ type: "lock_item", item_id: "note-1", locked: true }],
      },
    });
    const snap4 = await mockGetBoardSnapshot({ projectId });
    expect(snap4.items[0].locked).toBe(true);

    // Moving locked item is rejected
    const moveLockedResult = await mockApplyBoardCommands({
      projectId,
      commandBatch: {
        base_version: 4,
        commands: [{ type: "move_item", item_id: "note-1", position: { x: 999, y: 999 } }],
      },
    });
    expect(moveLockedResult.status).toBe("rejected");
    expect(moveLockedResult.diagnostics).toHaveLength(1);

    // Unlock item and remove it
    await mockApplyBoardCommands({
      projectId,
      commandBatch: {
        base_version: 4,
        commands: [
          { type: "lock_item", item_id: "note-1", locked: false },
          { type: "remove_item", item_id: "note-1" },
        ],
      },
    });
    const snap5 = await mockGetBoardSnapshot({ projectId });
    expect(snap5.items).toHaveLength(0);

    unsubscribe();
  });

  test("version conflict rejects batch without modifying version or items", async () => {
    // Current version is 0; provide base_version: 99
    const conflictResult = await mockApplyBoardCommands({
      projectId,
      commandBatch: {
        base_version: 99,
        commands: [
          {
            type: "add_item",
            item_id: "note-x",
            kind: { type: "note", content: "" },
            position: { x: 0, y: 0 },
            size: { width: 100, height: 100 },
          },
        ],
      },
    });

    expect(conflictResult.status).toBe("rejected");
    expect(conflictResult.diagnostics[0]).toMatchObject({
      code: "CORE/BOARD_VERSION_CONFLICT",
    });

    const snap = await mockGetBoardSnapshot({ projectId });
    expect(snap.version).toBe(0);
    expect(snap.items).toHaveLength(0);
  });

  test("undo and redo restore items and advance document version", async () => {
    // Undo on empty history returns null
    const nullUndo = await mockUndoBoard({ projectId });
    expect(nullUndo).toBeNull();

    // 1. Add item 1
    await mockApplyBoardCommands({
      projectId,
      commandBatch: {
        base_version: 0,
        commands: [
          {
            type: "add_item",
            item_id: "note-1",
            kind: { type: "note", content: "first" },
            position: { x: 0, y: 0 },
            size: { width: 100, height: 100 },
          },
        ],
      },
    });

    // 2. Add item 2
    await mockApplyBoardCommands({
      projectId,
      commandBatch: {
        base_version: 1,
        commands: [
          {
            type: "add_item",
            item_id: "note-2",
            kind: { type: "note", content: "second" },
            position: { x: 10, y: 10 },
            size: { width: 100, height: 100 },
          },
        ],
      },
    });

    const snapBeforeUndo = await mockGetBoardSnapshot({ projectId });
    expect(snapBeforeUndo.items).toHaveLength(2);
    expect(snapBeforeUndo.version).toBe(2);

    // 3. Undo: removes note-2, advances version to 3
    const undoResult = await mockUndoBoard({ projectId });
    expect(undoResult?.status).toBe("applied");
    expect(undoResult?.version).toBe(3);

    const snapAfterUndo = await mockGetBoardSnapshot({ projectId });
    expect(snapAfterUndo.items).toHaveLength(1);
    expect(snapAfterUndo.items[0].id).toBe("note-1");

    // 4. Redo: restores note-2, advances version to 4
    const redoResult = await mockRedoBoard({ projectId });
    expect(redoResult?.status).toBe("applied");
    expect(redoResult?.version).toBe(4);

    const snapAfterRedo = await mockGetBoardSnapshot({ projectId });
    expect(snapAfterRedo.items).toHaveLength(2);

    // 5. Redo when no future history returns null
    const nullRedo = await mockRedoBoard({ projectId });
    expect(nullRedo).toBeNull();
  });
  test("undo and redo reject a stale expected entry before mutating", async () => {
    const first = await mockApplyBoardCommands({
      projectId,
      commandBatch: {
        id: "history-first",
        base_version: 0,
        commands: [
          {
            type: "add_item",
            item_id: "note-1",
            kind: { type: "note", content: "first" },
            position: { x: 0, y: 0 },
            size: { width: 100, height: 100 },
          },
        ],
      },
    });
    const second = await mockApplyBoardCommands({
      projectId,
      commandBatch: {
        id: "history-second",
        base_version: 1,
        commands: [
          {
            type: "add_item",
            item_id: "note-2",
            kind: { type: "note", content: "second" },
            position: { x: 10, y: 10 },
            size: { width: 100, height: 100 },
          },
        ],
      },
    });
    const firstId = first.historyEntryId!;
    const secondId = second.historyEntryId!;

    await expect(mockUndoBoard({ projectId, expectedEntryId: firstId })).rejects.toMatchObject({
      code: "conflict",
      details: {
        project_id: projectId,
        action: "undo",
        expected_entry_id: firstId,
        undo_entry_id: secondId,
        redo_entry_id: null,
      },
    });

    // The rejected request changed neither the board nor its history.
    let snapshot = await mockGetBoardSnapshot({ projectId });
    expect(snapshot.items.map((item) => item.id)).toEqual(["note-1", "note-2"]);
    expect(snapshot.version).toBe(2);
    let page = await mockOpInvoke({
      projectId,
      name: "activity.list",
      input: { limit: 50 },
    });
    expect(
      (page.output as { records: Array<{ kind: string }> }).records.map((r) => r.kind),
    ).toEqual(["applied", "applied"]);

    await mockUndoBoard({ projectId, expectedEntryId: secondId });
    snapshot = await mockGetBoardSnapshot({ projectId });
    expect(snapshot.items.map((item) => item.id)).toEqual(["note-1"]);
    expect(snapshot.version).toBe(3);

    await expect(mockRedoBoard({ projectId, expectedEntryId: firstId })).rejects.toMatchObject({
      code: "conflict",
      details: {
        project_id: projectId,
        action: "redo",
        expected_entry_id: firstId,
        undo_entry_id: firstId,
        redo_entry_id: secondId,
      },
    });
    snapshot = await mockGetBoardSnapshot({ projectId });
    expect(snapshot.items.map((item) => item.id)).toEqual(["note-1"]);
    expect(snapshot.version).toBe(3);
    page = await mockOpInvoke({
      projectId,
      name: "activity.list",
      input: { limit: 50 },
    });
    expect(
      (page.output as { records: Array<{ kind: string }> }).records.map((r) => r.kind),
    ).toEqual(["undone", "applied", "applied"]);

    const redone = await mockRedoBoard({ projectId, expectedEntryId: secondId });
    expect(redone?.historyEntryId).toBe(secondId);
  });

  test("missing item error matches Rust code and message", async () => {
    const res = await mockApplyBoardCommands({
      projectId,
      commandBatch: {
        base_version: 0,
        commands: [{ type: "move_item", item_id: "missing-1", position: { x: 10, y: 10 } }],
      },
    });
    expect(res.status).toBe("rejected");
    expect(res.diagnostics[0]).toMatchObject({
      code: "CORE/BOARD_ITEM_MISSING",
      message: "board item does not exist",
      id: "board-item-missing-missing-1",
    });
  });

  test("commands that change nothing produce no_op without advancing version, adding history, or emitting events", async () => {
    const events: Event[] = [];
    const unsubscribe = subscribeEvents((e) => events.push(e));

    // Add note-1
    await mockApplyBoardCommands({
      projectId,
      commandBatch: {
        base_version: 0,
        commands: [
          {
            type: "add_item",
            item_id: "note-1",
            kind: { type: "note", content: "hello" },
            position: { x: 10, y: 20 },
            size: { width: 100, height: 100 },
          },
        ],
      },
    });
    events.length = 0;

    // Send a batch of identical properties: same position, same size, same content, same z, same locked
    const noOpRes = await mockApplyBoardCommands({
      projectId,
      commandBatch: {
        base_version: 1,
        commands: [
          { type: "move_item", item_id: "note-1", position: { x: 10, y: 20 } },
          { type: "resize_item", item_id: "note-1", size: { width: 100, height: 100 } },
          { type: "update_note", item_id: "note-1", content: "hello" },
          { type: "set_z", item_id: "note-1", z: 0 },
          { type: "lock_item", item_id: "note-1", locked: false },
        ],
      },
    });

    expect(noOpRes.status).toBe("no_op");
    expect(noOpRes.version).toBe(1);
    expect(noOpRes.changes).toHaveLength(0);
    expect(noOpRes.historyEntryId).toBeNull();
    expect(events).toHaveLength(0);

    const snap = await mockGetBoardSnapshot({ projectId });
    expect(snap.version).toBe(1);

    // Verify history was not pushed: undo should undo the original add_item (restoring 0 items)
    const undoRes = await mockUndoBoard({ projectId });
    expect(undoRes?.status).toBe("applied");
    const snapAfterUndo = await mockGetBoardSnapshot({ projectId });
    expect(snapAfterUndo.items).toHaveLength(0);

    unsubscribe();
  });

  test("update_note on non-note item or locked item produces Rust diagnostics", async () => {
    // Add workflow_ref item
    await mockApplyBoardCommands({
      projectId,
      commandBatch: {
        base_version: 0,
        commands: [
          {
            type: "add_item",
            item_id: "wf-1",
            kind: { type: "workflow_ref", workflow_id: "wf-abc" },
            position: { x: 0, y: 0 },
            size: { width: 100, height: 100 },
          },
        ],
      },
    });

    // 1. update_note on workflow_ref
    const notNoteRes = await mockApplyBoardCommands({
      projectId,
      commandBatch: {
        base_version: 1,
        commands: [{ type: "update_note", item_id: "wf-1", content: "text" }],
      },
    });
    expect(notNoteRes.status).toBe("rejected");
    expect(notNoteRes.diagnostics[0]).toMatchObject({
      code: "CORE/BOARD_ITEM_NOT_NOTE",
      message: "board item is not a note",
    });

    // 2. Lock item and try update_note
    await mockApplyBoardCommands({
      projectId,
      commandBatch: {
        base_version: 1,
        commands: [{ type: "lock_item", item_id: "wf-1", locked: true }],
      },
    });

    const lockedRes = await mockApplyBoardCommands({
      projectId,
      commandBatch: {
        base_version: 2,
        commands: [{ type: "update_note", item_id: "wf-1", content: "text" }],
      },
    });
    expect(lockedRes.status).toBe("rejected");
    expect(lockedRes.diagnostics[0]).toMatchObject({
      code: "CORE/BOARD_ITEM_LOCKED",
      message: "cannot modify locked board item",
    });
  });
  test("mockDeleteProject clears that project's mock board and history via mockResetBoard", async () => {
    const { mockDeleteProject } = await import("../src/ipc/mock");
    await mockApplyBoardCommands({
      projectId: "proj-delete-test",
      commandBatch: {
        base_version: 0,
        commands: [
          {
            type: "add_item",
            item_id: "note-del",
            kind: { type: "note", content: "to delete" },
            position: { x: 0, y: 0 },
            size: { width: 100, height: 100 },
          },
        ],
      },
    });

    const snapBefore = await mockGetBoardSnapshot({ projectId: "proj-delete-test" });
    expect(snapBefore.items).toHaveLength(1);
    expect(snapBefore.version).toBe(1);

    await mockDeleteProject({ projectId: "proj-delete-test" });

    // After deleting project, snapshot is fresh/cleared
    const snapAfter = await mockGetBoardSnapshot({ projectId: "proj-delete-test" });
    expect(snapAfter.items).toHaveLength(0);
    expect(snapAfter.version).toBe(0);
  });

  test("apply, undo and redo append activity the list operation returns", async () => {
    const events: Event[] = [];
    const unsubscribe = subscribeEvents((e) => events.push(e));

    const applied = await mockApplyBoardCommands({
      projectId,
      commandBatch: {
        id: "activity-batch",
        base_version: 0,
        commands: [
          {
            type: "add_item",
            item_id: "note-activity",
            kind: { type: "note", content: "feed me" },
            position: { x: 0, y: 0 },
            size: { width: 100, height: 100 },
          },
        ],
      },
    });
    expect(applied.historyEntryId).toBe("board-history:activity-batch");

    const appended = events.filter((e) => e.type === "mod" && e.kind === "activity.appended");
    expect(appended).toHaveLength(1);
    const payload = (appended[0] as { payload: unknown }).payload as {
      projectId: string;
      record: Record<string, unknown>;
    };
    expect(payload.projectId).toBe(projectId);
    expect(payload.record).toMatchObject({
      kind: "applied",
      summaryKey: "activity.board.addNote",
      summaryParams: { count: 1 },
      targets: ["note-activity"],
      historyEntryId: "board-history:activity-batch",
    });

    const undone = await mockUndoBoard({ projectId });
    expect(undone?.historyEntryId).toBe("board-history:activity-batch");
    const redone = await mockRedoBoard({ projectId });
    expect(redone?.historyEntryId).toBe("board-history:activity-batch");

    const page = await mockOpInvoke({
      projectId,
      name: "activity.list",
      input: { limit: 50 },
    });
    const { records, has_more } = page.output as {
      records: Array<Record<string, unknown>>;
      has_more: boolean;
    };
    expect(records.map((record) => record.kind)).toEqual(["redone", "undone", "applied"]);
    expect(has_more).toBe(false);

    unsubscribe();
  });
});
