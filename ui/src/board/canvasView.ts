import type { Viewport } from "@xyflow/react";
import { LAYOUT } from "@/design/tokens";

/**
 * Width the floating session panel covers on the right, including the gap on
 * both of its sides. The canvas instruments and overlays are centered in the
 * remaining area so they never sit underneath the panel.
 */
export const SESSION_PANEL_COVERAGE = LAYOUT.panel + LAYOUT.panelGap * 2;

export const CANVAS_MIN_ZOOM = 0.2;
export const CANVAS_MAX_ZOOM = 2.5;

/** Instruments slide out of the panel's way at the shell's slow tempo. */
export const CANVAS_HUD_SHIFT_MS = 360;
/** Button-driven zoom and fit. */
export const CANVAS_VIEW_MS = 250;
/** Programmatic focus on one item, used by the library. */
export const CANVAS_FOCUS_MS = 300;
/** Distance from the canvas edge to each floating plate. */
export const CANVAS_HUD_EDGE = 20;

/** Gap between the zoom plate and the dock when the plates are stacked. */
export const CANVAS_HUD_STACK_GAP = 8;

/**
 * Tool dock height before it has been measured: a 32px button row plus the
 * plate's 8px padding. The live height is fed back into `hudLayout`.
 */
export const CANVAS_HUD_DOCK_HEIGHT = 40;

/**
 * Free-strip width below which the zoom plate moves above the dock.
 *
 * The plates measure about 131px (dock) and 146px (zoom) wide at the default
 * type scale. With the dock centered, its right edge sits at
 * `freeWidth / 2 + 65.5`; the zoom plate's left edge sits at
 * `freeWidth - 146` (or `- 20` with the panel closed). A 520px strip leaves
 * roughly 48px between them at the switch, which covers the measured plates,
 * the 20px edge inset and breathing room for future buttons. The 1024x640
 * minimum with the expanded sidebar and the panel open leaves 356px and
 * stacks; the 1440 default leaves 772px and stays wide.
 */
export const CANVAS_HUD_STACK_THRESHOLD = 520;

export type HudLayout = {
  /** True when the zoom plate sits above the dock instead of beside it. */
  stacked: boolean;
  /** Left edge of the centered dock, in px from the canvas's left edge. */
  dockLeft: number;
  /** Right edge of the zoom plate, in px from the canvas's right edge. */
  zoomRight: number;
  /** Bottom edge of the zoom plate, in px from the canvas's bottom edge. */
  zoomBottom: number;
};

export type HudLayoutInput = {
  /** Canvas width left visible by the session panel. */
  freeWidth: number;
  /** Distance from the canvas's right edge to the visible strip's right edge. */
  rightInset?: number;
  /** Rendered dock height, used only by the stacked arrangement. */
  dockHeight?: number;
};

/**
 * Places the two floating plates inside the strip the session panel leaves
 * free. The dock stays centered; the zoom plate keeps the strip's right edge
 * and only moves up when the strip is too narrow to hold both plates.
 */
export function hudLayout({
  freeWidth,
  rightInset = CANVAS_HUD_EDGE,
  dockHeight = CANVAS_HUD_DOCK_HEIGHT,
}: HudLayoutInput): HudLayout {
  const strip = Math.max(0, freeWidth);
  const stacked = strip < CANVAS_HUD_STACK_THRESHOLD;
  return {
    stacked,
    dockLeft: strip / 2,
    zoomRight: Math.max(0, rightInset),
    zoomBottom: stacked ? CANVAS_HUD_EDGE + dockHeight + CANVAS_HUD_STACK_GAP : CANVAS_HUD_EDGE,
  };
}

function cubicBezier(x1: number, y1: number, x2: number, y2: number) {
  const cx = 3 * x1;
  const bx = 3 * (x2 - x1) - cx;
  const ax = 1 - cx - bx;
  const cy = 3 * y1;
  const by = 3 * (y2 - y1) - cy;
  const ay = 1 - cy - by;
  const sampleX = (t: number) => ((ax * t + bx) * t + cx) * t;
  const sampleY = (t: number) => ((ay * t + by) * t + cy) * t;
  const derivativeX = (t: number) => (3 * ax * t + 2 * bx) * t + cx;

  return (x: number) => {
    let t = x;
    for (let i = 0; i < 5; i += 1) {
      const error = sampleX(t) - x;
      if (Math.abs(error) < 1e-5) break;
      const derivative = derivativeX(t);
      if (Math.abs(derivative) < 1e-6) break;
      t -= error / derivative;
    }
    return sampleY(Math.min(1, Math.max(0, t)));
  };
}

/** The shared `--ease-ui` curve: cubic-bezier(0.32, 0.72, 0, 1). */
export const easeUi = cubicBezier(0.32, 0.72, 0, 1);

function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

/**
 * Interpolates a viewport with the shell easing on the animation frame clock.
 * Returns a cancel function; callers own at most one animation at a time.
 */
export function animateViewport(
  from: Viewport,
  to: Viewport,
  apply: (viewport: Viewport) => void,
  duration = CANVAS_VIEW_MS,
): () => void {
  if (duration <= 0 || prefersReducedMotion()) {
    apply(to);
    return () => {};
  }

  const startedAt = performance.now();
  let frame = 0;

  const step = (now: number) => {
    const progress = Math.min(1, (now - startedAt) / duration);
    if (progress >= 1) {
      apply(to);
      return;
    }
    const k = easeUi(progress);
    apply({
      x: from.x + (to.x - from.x) * k,
      y: from.y + (to.y - from.y) * k,
      zoom: from.zoom + (to.zoom - from.zoom) * k,
    });
    frame = requestAnimationFrame(step);
  };

  frame = requestAnimationFrame(step);
  return () => cancelAnimationFrame(frame);
}
