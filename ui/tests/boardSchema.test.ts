import { describe, test, expect, vi } from "vitest";
import { BoardItemKindSchema, BoardItemSchema, BoardSnapshotSchema } from "../src/ipc/schemas";

describe("BoardItemKind and BoardSnapshot schemas", () => {
  test("parses the five known Rust kinds", () => {
    const wf = BoardItemKindSchema.parse({
      type: "workflow_ref",
      workflow_id: "wf-100",
    });
    expect(wf).toEqual({ type: "workflow_ref", workflow_id: "wf-100" });

    const asset = BoardItemKindSchema.parse({
      type: "asset_ref",
      artifact_id: "art-200",
    });
    expect(asset).toEqual({ type: "asset_ref", artifact_id: "art-200" });

    const note = BoardItemKindSchema.parse({
      type: "note",
      content: "Hello world",
    });
    expect(note).toEqual({ type: "note", content: "Hello world" });

    const run = BoardItemKindSchema.parse({
      type: "run_ref",
      run_id: "run-300",
    });
    expect(run).toEqual({ type: "run_ref", run_id: "run-300" });

    const col = BoardItemKindSchema.parse({
      type: "collection",
      collection_id: "col-400",
    });
    expect(col).toEqual({ type: "collection", collection_id: "col-400" });
  });

  test("unknown kind parses in BoardItemSchema without failing", () => {
    const item = BoardItemSchema.parse({
      id: "item-custom",
      kind: {
        type: "audio_track",
        duration: 120,
        metadata: { bitRate: 320 },
      },
      position: { x: 50, y: 100 },
      size: { width: 200, height: 100 },
      z: 1,
      locked: false,
    });

    expect(item.id).toBe("item-custom");
    expect(item.kind.type).toBe("audio_track");
    expect((item.kind as { duration: number }).duration).toBe(120);
  });

  test("BoardSnapshotSchema parses snapshot with all five known kinds and an unknown kind", () => {
    const snapshot = BoardSnapshotSchema.parse({
      id: "board-main",
      projectId: "proj-1",
      version: 5,
      items: [
        {
          id: "item-1",
          kind: { type: "workflow_ref", workflow_id: "wf-1" },
          position: { x: 0, y: 0 },
          size: { width: 200, height: 100 },
          z: 0,
          locked: false,
        },
        {
          id: "item-2",
          kind: { type: "asset_ref", artifact_id: "art-1" },
          position: { x: 250, y: 0 },
          size: { width: 200, height: 100 },
          z: 1,
          locked: false,
        },
        {
          id: "item-3",
          kind: { type: "note", content: "My note" },
          position: { x: 500, y: 0 },
          size: { width: 200, height: 100 },
          z: 2,
          locked: false,
        },
        {
          id: "item-4",
          kind: { type: "run_ref", run_id: "run-1" },
          position: { x: 0, y: 150 },
          size: { width: 200, height: 100 },
          z: 3,
          locked: false,
        },
        {
          id: "item-5",
          kind: { type: "collection", collection_id: "col-1" },
          position: { x: 250, y: 150 },
          size: { width: 200, height: 100 },
          z: 4,
          locked: false,
        },
        {
          id: "item-6",
          kind: { type: "future_plugin_item", setting: true },
          position: { x: 500, y: 150 },
          size: { width: 200, height: 100 },
          z: 5,
          locked: true,
        },
      ],
    });

    expect(snapshot.items).toHaveLength(6);
    expect(snapshot.version).toBe(5);
  });

  test("missing type in kind fails validation", () => {
    expect(() =>
      BoardItemSchema.parse({
        id: "bad-item",
        kind: { notype: "here" },
        position: { x: 0, y: 0 },
        size: { width: 100, height: 100 },
        z: 0,
        locked: false,
      }),
    ).toThrow();
  });

  test("malformed known kind fails validation and does not fall through to unknown kind fallback", () => {
    // note without content
    expect(() =>
      BoardItemSchema.parse({
        id: "bad-note",
        kind: { type: "note" },
        position: { x: 0, y: 0 },
        size: { width: 100, height: 100 },
        z: 0,
        locked: false,
      }),
    ).toThrow();

    // workflow_ref without workflow_id
    expect(() =>
      BoardItemSchema.parse({
        id: "bad-wf",
        kind: { type: "workflow_ref" },
        position: { x: 0, y: 0 },
        size: { width: 100, height: 100 },
        z: 0,
        locked: false,
      }),
    ).toThrow();
  });

  test("getBoardSnapshot validates response against BoardSnapshotSchema and throws on malformed snapshot", async () => {
    const { getBoardSnapshot } = await import("../src/ipc/commands");
    // Valid snapshot succeeds
    const snap = await getBoardSnapshot("valid-proj");
    expect(snap.projectId).toBe("valid-proj");

    // Malformed snapshot from IPC fails BoardSnapshotSchema validation
    const mockModule = await import("../src/ipc/mock");
    vi.spyOn(mockModule, "mockGetBoardSnapshot").mockResolvedValueOnce({
      id: "board-invalid",
      projectId: "valid-proj",
      version: "not-a-number" as unknown as number,
      items: [],
    });
    await expect(getBoardSnapshot("valid-proj")).rejects.toThrow();
  });
});
