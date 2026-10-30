import { describe, test, expect } from "vitest";
import { boardNodeTraceClass } from "../src/board/trace";
import type { CanvasTrace } from "../src/store/uiStore";

function trace(actorKind: CanvasTrace["actorKind"], targets = ["note-1"]): CanvasTrace {
  return { targets, actorKind };
}

describe("boardNodeTraceClass", () => {
  test("no trace leaves the wrapper untouched", () => {
    expect(boardNodeTraceClass(null, "note-1")).toBe("");
  });

  test("targets of a human record get the accent ring", () => {
    const className = boardNodeTraceClass(trace("Human"), "note-1");
    expect(className).toContain("shadow-[0_0_0_2px_var(--color-accent)]");
    expect(className).toContain("rounded-lg");
    expect(className).not.toContain("opacity-40");
  });

  test("agent targets use gold instead of the accent", () => {
    const className = boardNodeTraceClass(trace("Agent"), "note-1");
    expect(className).toContain("shadow-[0_0_0_2px_var(--color-agent)]");
    expect(className).not.toContain("var(--color-accent)");
  });

  test("importer and system records keep the accent ring", () => {
    for (const kind of ["Importer", "System"] as const) {
      expect(boardNodeTraceClass(trace(kind), "note-1")).toContain("var(--color-accent)");
    }
  });

  test("items outside the target list fade and get no ring", () => {
    const className = boardNodeTraceClass(trace("Human"), "note-other");
    expect(className).toBe("opacity-40");
  });

  test("a trace without targets dims everything", () => {
    expect(boardNodeTraceClass(trace("Human", []), "note-1")).toBe("opacity-40");
  });
});
