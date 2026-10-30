import { expect, test } from "vitest";
import { BoardSnapshotSchema, DocumentEventSchema, ProjectSchema } from "../src/ipc/schemas";

test("project and board DTOs preserve stable camelCase fields", () => {
  const project = ProjectSchema.parse({
    id: "p1",
    name: "Project",
    description: "desc",
    createdAt: "a",
    updatedAt: "b",
  });
  expect(project.id).toBe("p1");
  const board = BoardSnapshotSchema.parse({
    id: "board-p1",
    projectId: "p1",
    version: 4,
    items: [],
  });
  expect(board.version).toBe(4);
});

test("document events require project/document identity and version", () => {
  expect(
    DocumentEventSchema.parse({
      type: "document",
      kind: "workflow.changed",
      projectId: "p1",
      documentId: "wf1",
      version: 2,
    }),
  ).toEqual({
    type: "document",
    kind: "workflow.changed",
    projectId: "p1",
    documentId: "wf1",
    version: 2,
  });
  expect(() =>
    DocumentEventSchema.parse({ type: "document", kind: "workflow.changed", projectId: "p1" }),
  ).toThrow();
});
