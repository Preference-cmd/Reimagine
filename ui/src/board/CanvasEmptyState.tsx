import type { CSSProperties } from "react";
import { Link } from "@tanstack/react-router";
import * as m from "$paraglide/messages";
import { useUIStore } from "@/store/uiStore";
import { CANVAS_HUD_SHIFT_MS, SESSION_PANEL_COVERAGE } from "./canvasView";

const PRIMARY_ACTION =
  "inline-flex h-7 cursor-pointer items-center rounded-full bg-surface px-3 text-caption font-medium whitespace-nowrap text-fg shadow-[0_0_0_1px_var(--color-hair-strong),var(--shadow-1)] transition-[background-color,box-shadow,transform,color] duration-150 ease-ui hover:shadow-[0_0_0_1px_var(--color-hair-strong),var(--shadow-2)] active:scale-[0.97] focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none";
const GHOST_ACTION =
  "inline-flex h-7 cursor-pointer items-center rounded-full bg-transparent px-3 text-caption font-medium whitespace-nowrap text-fg2 transition-[background-color,color,transform] duration-150 ease-ui hover:bg-hover hover:text-fg active:scale-[0.97] focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none";

type CanvasEmptyStateProps = {
  onCreateNote: () => void;
};

/**
 * Empty-canvas overlay. It is centered in the area the session panel leaves
 * free, lets pointer events through to the canvas except on its own card, and
 * is only mounted once a snapshot has loaded.
 */
export function CanvasEmptyState({ onCreateNote }: CanvasEmptyStateProps) {
  const panelOpen = useUIStore((state) => state.sessionPanelOpen);

  const style: CSSProperties = {
    right: panelOpen ? SESSION_PANEL_COVERAGE : 0,
    transition: `right ${CANVAS_HUD_SHIFT_MS}ms var(--ease-ui)`,
  };

  return (
    <div
      className="pointer-events-none absolute inset-y-0 left-0 z-[5] grid place-items-center motion-reduce:transition-none"
      style={style}
      data-testid="canvas-empty-state"
    >
      <div className="pointer-events-auto max-w-[340px] px-6 text-center">
        <h2 className="text-[15px] font-semibold tracking-[-0.01em] text-fg">
          {m["board.emptyTitle"]()}
        </h2>
        <p className="mt-1.5 mb-4 text-ui leading-[1.65] text-fg3">
          {m["board.emptyDescription"]()}
        </p>
        <div className="flex justify-center gap-1.5">
          <Link to="/library" className={PRIMARY_ACTION}>
            {m["board.openLibrary"]()}
          </Link>
          <button type="button" onClick={onCreateNote} className={GHOST_ACTION}>
            {m["board.newNote"]()}
          </button>
        </div>
      </div>
    </div>
  );
}
