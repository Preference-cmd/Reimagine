import { TopBar } from "./TopBar";
import { SideRail } from "./SideRail";
import { ExplorerPanel } from "./ExplorerPanel";
import { PropertiesPanel } from "./PropertiesPanel";
import { NodeCanvas } from "@/components/canvas/NodeCanvas";

/**
 * AppShell — root layout for the editor workspace.
 *
 * Structure (mirrors docs/design/editor/ref.html):
 *   - TopBar: 56px overlay at the top, transparent + blur, floats over canvas
 *   - SideRail: 64px overlay on the left, full height below top
 *   - main: fills the viewport region to the right of the SideRail; the
 *     canvas extends under the floating TopBar (so the TopBar's
 *     backdrop-blur has something to sample)
 *     - NodeCanvas: React Flow with demo nodes (issue 05)
 *     - ExplorerPanel / PropertiesPanel: floating glass panes positioned
 *       absolutely inside main, so they automatically track viewport
 *       resizes
 *
 * Layout uses a CSS grid to keep the main column honest on resize:
 *   columns: [64px sidebar track | 1fr main track]
 *   row:     1fr (= full viewport height)
 *
 * TopBar and SideRail are `position: absolute` (not `fixed`) so their width
 * is measured against this shell — which is sized to the scaled #root (see
 * globals.css) — and the 0.7 transform on #root paints them at full
 * viewport width. The grid only contains main as a real grid item.
 */
export function AppShell() {
  return (
    <div className="relative grid h-full w-full grid-cols-[64px_1fr] grid-rows-[1fr] bg-background text-foreground">
      <TopBar />
      <SideRail />
      <main className="relative col-start-2 overflow-hidden">
        <NodeCanvas />
        <ExplorerPanel />
        <PropertiesPanel />
      </main>
    </div>
  );
}