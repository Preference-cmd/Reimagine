import { expect, test } from "vitest";
import { BoardSnapshotSchema, DocumentChangedEventSchema, ProjectSchema } from "../src/ipc/schemas";
import { workflowFromJson, workflowToJson } from "../src/lib/workflowCodec";

test("project and board DTOs preserve stable camelCase fields", () => {
  const project = ProjectSchema.parse({ id: "p1", name: "Project", description: "desc", createdAt: "a", updatedAt: "b" });
  expect(project.id).toBe("p1");
  const board = BoardSnapshotSchema.parse({ id: "board-p1", projectId: "p1", version: 4, items: [] });
  expect(board.version).toBe(4);
});

test("document events require project/document identity and version", () => {
  expect(DocumentChangedEventSchema.parse({ kind: "workflow.changed", projectId: "p1", documentId: "wf1", version: 2 })).toEqual({ kind: "workflow.changed", projectId: "p1", documentId: "wf1", version: 2 });
  expect(() => DocumentChangedEventSchema.parse({ kind: "workflow.changed", projectId: "p1" })).toThrow();
});

test("workflow codec round-trips a supplied document version", () => {
  const json = workflowToJson([], [], "wf1", "Workflow", 7);
  expect(json.version).toBe(7);
  const graph = workflowFromJson(json);
  expect(graph.version).toBe(7);
  expect(graph.name).toBe("Workflow");
});