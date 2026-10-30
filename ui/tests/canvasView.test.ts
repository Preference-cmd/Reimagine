import { describe, expect, it } from "vitest";
import {
  CANVAS_HUD_DOCK_HEIGHT,
  CANVAS_HUD_EDGE,
  CANVAS_HUD_STACK_GAP,
  CANVAS_HUD_STACK_THRESHOLD,
  hudLayout,
  SESSION_PANEL_COVERAGE,
} from "../src/board/canvasView";

describe("hudLayout", () => {
  it("keeps the wide arrangement at the 1440 default with the panel open", () => {
    const layout = hudLayout({
      freeWidth: 772,
      rightInset: SESSION_PANEL_COVERAGE,
    });

    expect(layout).toEqual({
      stacked: false,
      dockLeft: 386,
      zoomRight: SESSION_PANEL_COVERAGE,
      zoomBottom: CANVAS_HUD_EDGE,
    });
  });

  it("stacks the zoom plate on the dock's height plus the gap on a narrow strip", () => {
    const layout = hudLayout({
      freeWidth: 356,
      rightInset: SESSION_PANEL_COVERAGE,
      dockHeight: 48,
    });

    expect(layout.stacked).toBe(true);
    expect(layout.dockLeft).toBe(178);
    expect(layout.zoomRight).toBe(SESSION_PANEL_COVERAGE);
    expect(layout.zoomBottom).toBe(CANVAS_HUD_EDGE + 48 + CANVAS_HUD_STACK_GAP);
  });

  it("falls back to the measured dock height and the canvas edge inset", () => {
    const layout = hudLayout({ freeWidth: 356 });

    expect(layout.stacked).toBe(true);
    expect(layout.zoomRight).toBe(CANVAS_HUD_EDGE);
    expect(layout.zoomBottom).toBe(CANVAS_HUD_EDGE + CANVAS_HUD_DOCK_HEIGHT + CANVAS_HUD_STACK_GAP);
  });

  it("switches arrangement exactly at the threshold", () => {
    const below = hudLayout({ freeWidth: CANVAS_HUD_STACK_THRESHOLD - 1 });
    const at = hudLayout({ freeWidth: CANVAS_HUD_STACK_THRESHOLD });

    expect(below.stacked).toBe(true);
    expect(at.stacked).toBe(false);
  });

  it("keeps the dock centered and never returns a negative strip", () => {
    expect(hudLayout({ freeWidth: 0 }).dockLeft).toBe(0);
    expect(hudLayout({ freeWidth: -40 }).dockLeft).toBe(0);
  });
});
