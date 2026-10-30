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
