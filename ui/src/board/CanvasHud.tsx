import { useCallback, useEffect, useRef, type CSSProperties, type ReactNode } from "react";
import {
  getNodesBounds,
  getViewportForBounds,
  useReactFlow,
  useStore,
  useViewport,
  type Viewport,
} from "@xyflow/react";
import { File as FileIcon, Hand, Minus, MousePointer2, Plus, Scan } from "lucide-react";
import * as m from "$paraglide/messages";
import { cn } from "@/lib/utils";
import { ICON_STROKE } from "@/design/tokens";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useUIStore } from "@/store/uiStore";
import {
  animateViewport,
  CANVAS_HUD_EDGE,
  CANVAS_HUD_SHIFT_MS,
  CANVAS_MAX_ZOOM,
  CANVAS_MIN_ZOOM,
  CANVAS_VIEW_MS,
  SESSION_PANEL_COVERAGE,
} from "./canvasView";

export type CanvasTool = "select" | "hand" | "note";

const PLATE =
  "pointer-events-auto flex items-center gap-0.5 rounded-[14px] bg-surface p-1 shadow-[var(--shadow-bezel),var(--shadow-2),var(--shadow-hi)]";

const TOOL_BUTTON =
  "flex h-8 w-9 shrink-0 cursor-pointer items-center justify-center rounded-md text-fg2 transition-[background-color,color,box-shadow,transform] duration-150 ease-ui active:scale-[0.94] active:bg-press focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none";
const TOOL_ACTIVE = "bg-accent-tint text-accent shadow-[inset_0_0_0_1px_var(--color-accent-line)]";

const ZOOM_BUTTON =
  "flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-md text-fg2 transition-[background-color,color,transform] duration-150 ease-ui hover:bg-hover hover:text-fg active:scale-[0.94] active:bg-press focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none";

type HudButtonProps = {
  label: string;
  hint?: string;
  pressed?: boolean;
  onClick: () => void;
  className?: string;
  children: ReactNode;
};

function HudButton({ label, hint, pressed, onClick, className, children }: HudButtonProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={label}
          aria-pressed={pressed}
          onClick={onClick}
          className={cn(
            TOOL_BUTTON,
            pressed ? TOOL_ACTIVE : "hover:bg-hover hover:text-fg",
            className,
          )}
        >
          {children}
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" sideOffset={8}>
        {hint ?? label}
      </TooltipContent>
    </Tooltip>
  );
}

type CanvasHudProps = {
  tool: CanvasTool;
  onToolChange: (tool: CanvasTool) => void;
};

/**
 * Floating canvas instruments: the bottom-center tool dock and the bottom-right
 * zoom bar. Both plates shift left of the floating session panel while it is
 * open, so neither is ever covered.
 */
export function CanvasHud({ tool, onToolChange }: CanvasHudProps) {
  const panelOpen = useUIStore((state) => state.sessionPanelOpen);
  const { zoom } = useViewport();
  const { getViewport, setViewport, getNodes } = useReactFlow();
  const width = useStore((state) => state.width);
  const height = useStore((state) => state.height);

  const animationRef = useRef<(() => void) | null>(null);
  useEffect(() => () => animationRef.current?.(), []);

  const visibleWidth = Math.max(1, width - (panelOpen ? SESSION_PANEL_COVERAGE : 0));

  const animateTo = useCallback(
    (next: Viewport) => {
      animationRef.current?.();
      animationRef.current = animateViewport(getViewport(), next, setViewport, CANVAS_VIEW_MS);
    },
    [getViewport, setViewport],
  );

  // Zooms around the center of the unobstructed area, not the panel-covered
  // pane center, so the focused content stays in view.
  const zoomToVisibleCenter = useCallback(
    (nextZoom: number) => {
      if (width <= 1 || height <= 1) return;
      const from = getViewport();
      const clamped = Math.min(CANVAS_MAX_ZOOM, Math.max(CANVAS_MIN_ZOOM, nextZoom));
      if (Math.abs(clamped - from.zoom) < 1e-4) return;
      const anchorX = visibleWidth / 2;
      const anchorY = height / 2;
      const scale = clamped / from.zoom;
      animateTo({
        x: anchorX - (anchorX - from.x) * scale,
        y: anchorY - (anchorY - from.y) * scale,
        zoom: clamped,
      });
    },
    [animateTo, getViewport, height, visibleWidth, width],
  );

  const handleZoomIn = useCallback(
    () => zoomToVisibleCenter(getViewport().zoom * 1.2),
    [getViewport, zoomToVisibleCenter],
  );
  const handleZoomOut = useCallback(
    () => zoomToVisibleCenter(getViewport().zoom / 1.2),
    [getViewport, zoomToVisibleCenter],
  );
  const handleResetZoom = useCallback(() => zoomToVisibleCenter(1), [zoomToVisibleCenter]);

  const handleFit = useCallback(() => {
    if (width <= 1 || height <= 1) return;
    const nodes = getNodes();
    if (nodes.length === 0) {
      animateTo({ x: visibleWidth / 2, y: height / 2, zoom: 1 });
      return;
    }
    const bounds = getNodesBounds(nodes);
    animateTo(getViewportForBounds(bounds, visibleWidth, height, CANVAS_MIN_ZOOM, 1, "40px"));
  }, [animateTo, getNodes, height, visibleWidth, width]);

  const dockStyle: CSSProperties = {
    left: panelOpen ? `calc((100% - ${SESSION_PANEL_COVERAGE}px) / 2)` : "50%",
    transition: `left ${CANVAS_HUD_SHIFT_MS}ms var(--ease-ui)`,
  };
  const zoomStyle: CSSProperties = {
    right: panelOpen ? SESSION_PANEL_COVERAGE : CANVAS_HUD_EDGE,
    transition: `right ${CANVAS_HUD_SHIFT_MS}ms var(--ease-ui)`,
  };

  return (
    <>
      <div
        role="toolbar"
        aria-label={m["board.canvasTools"]()}
        className={cn(
          PLATE,
          "absolute bottom-5 z-10 -translate-x-1/2 motion-reduce:transition-none",
        )}
        style={dockStyle}
        data-testid="canvas-dock"
      >
        <HudButton
          label={m["board.toolSelect"]()}
          hint={m["board.toolSelectHint"]()}
          pressed={tool === "select"}
          onClick={() => onToolChange("select")}
        >
          <MousePointer2 className="size-4" strokeWidth={ICON_STROKE} aria-hidden="true" />
        </HudButton>
        <HudButton
          label={m["board.toolHand"]()}
          hint={m["board.toolHandHint"]()}
          pressed={tool === "hand"}
          onClick={() => onToolChange("hand")}
        >
          <Hand className="size-4" strokeWidth={ICON_STROKE} aria-hidden="true" />
        </HudButton>
        <span aria-hidden="true" className="mx-1 h-[18px] w-px bg-hair-strong" />
        <HudButton
          label={m["board.toolNote"]()}
          hint={m["board.toolNoteHint"]()}
          pressed={tool === "note"}
          onClick={() => onToolChange("note")}
        >
          <FileIcon className="size-4" strokeWidth={ICON_STROKE} aria-hidden="true" />
        </HudButton>
      </div>

      <div
        role="group"
        aria-label={m["board.zoomControls"]()}
        className={cn(PLATE, "absolute bottom-5 z-10 motion-reduce:transition-none")}
        style={zoomStyle}
        data-testid="canvas-zoom"
      >
        <button
          type="button"
          aria-label={m["board.zoomOut"]()}
          onClick={handleZoomOut}
          className={ZOOM_BUTTON}
        >
          <Minus className="size-3.5" strokeWidth={ICON_STROKE} aria-hidden="true" />
        </button>
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              type="button"
              aria-label={m["board.zoomReset"]()}
              onClick={handleResetZoom}
              className="relative h-7 min-w-12 cursor-pointer rounded-md px-1 font-mono text-caption tabular-nums text-fg2 transition-colors duration-150 ease-ui hover:bg-hover hover:text-fg focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none"
            >
              {Math.round(zoom * 100)}%
            </button>
          </TooltipTrigger>
          <TooltipContent side="top" sideOffset={8}>
            {m["board.zoomReset"]()}
          </TooltipContent>
        </Tooltip>
        <button
          type="button"
          aria-label={m["board.zoomIn"]()}
          onClick={handleZoomIn}
          className={ZOOM_BUTTON}
        >
          <Plus className="size-3.5" strokeWidth={ICON_STROKE} aria-hidden="true" />
        </button>
        <button
          type="button"
          aria-label={m["board.zoomFit"]()}
          onClick={handleFit}
          className={ZOOM_BUTTON}
        >
          <Scan className="size-3.5" strokeWidth={ICON_STROKE} aria-hidden="true" />
        </button>
      </div>
    </>
  );
}
