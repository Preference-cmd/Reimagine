import { describe, expect, test } from "vitest";
import { canvasAreaLayout, placementCenter } from "../src/lib/canvasLayout";

describe("canvas area layout", () => {
  test("a 1440x900 window: panel open keeps a 772px strip, panel closed the full canvas", () => {
    const open = canvasAreaLayout({
      windowWidth: 1440,
      windowHeight: 900,
      sidebarCollapsed: false,
      panelOpen: true,
    });
    expect(open).toEqual({ width: 1168, height: 836, coverage: 396, visibleWidth: 772 });

    const closed = canvasAreaLayout({
      windowWidth: 1440,
      windowHeight: 900,
      sidebarCollapsed: false,
      panelOpen: false,
    });
    expect(closed).toEqual({ width: 1168, height: 836, coverage: 0, visibleWidth: 1168 });
  });

  test("a collapsed sidebar gives the canvas the sidebar width plus its left inset", () => {
    const area = canvasAreaLayout({
      windowWidth: 1440,
      windowHeight: 900,
      sidebarCollapsed: true,
      panelOpen: true,
    });
    expect(area.width).toBe(1424);
    expect(area.visibleWidth).toBe(1028);
    expect(area.height).toBe(836);
  });

  test("a panned viewport places at the flow point under the visible center", () => {
    const area = canvasAreaLayout({
      windowWidth: 1440,
      windowHeight: 900,
      sidebarCollapsed: false,
      panelOpen: true,
    });
    expect(placementCenter({ x: -600, y: -300, zoom: 1 }, area)).toEqual({ x: 986, y: 718 });
  });

  test("a cold start viewport centers at the middle of the visible strip", () => {
    const area = canvasAreaLayout({
      windowWidth: 1440,
      windowHeight: 900,
      sidebarCollapsed: false,
      panelOpen: true,
    });
    expect(placementCenter({ x: 0, y: 0, zoom: 1 }, area)).toEqual({ x: 386, y: 418 });
  });

  test("zoom divides the offset so the center follows the visible flow scale", () => {
    const area = canvasAreaLayout({
      windowWidth: 1440,
      windowHeight: 900,
      sidebarCollapsed: false,
      panelOpen: true,
    });
    expect(placementCenter({ x: -600, y: -300, zoom: 2 }, area)).toEqual({ x: 493, y: 359 });
  });

  test("the visible width never drops below one pixel", () => {
    const area = canvasAreaLayout({
      windowWidth: 1024,
      windowHeight: 640,
      sidebarCollapsed: false,
      panelOpen: true,
    });
    expect(area.width).toBe(752);
    expect(area.height).toBe(576);
    expect(area.visibleWidth).toBe(356);

    const tiny = canvasAreaLayout({
      windowWidth: 600,
      windowHeight: 600,
      sidebarCollapsed: false,
      panelOpen: true,
    });
    expect(tiny.visibleWidth).toBe(1);
  });
});
