import { Library } from "lucide-react";
import * as m from "$paraglide/messages";
import { ICON_STROKE } from "@/design/tokens";
import { useUIStore } from "@/store/uiStore";

/**
 * Library mode placeholder. Phase 2 replaces the empty state with the
 * workspace table (collections, dynamic workflows, runs, files).
 */
export function LibraryView() {
  const panelOpen = useUIStore((state) => state.sessionPanelOpen);

  return (
    <div
      className="flex h-full min-h-0 flex-col overflow-y-auto bg-surface px-6 pt-5 pb-12 transition-[padding] duration-300 ease-ui"
      style={panelOpen ? { paddingRight: "calc(var(--spacing-panel) + 48px)" } : undefined}
    >
      <header className="shrink-0">
        <h1 className="text-headline-md font-semibold tracking-[-0.02em] text-fg">
          {m["library.title"]()}
        </h1>
        <p className="mt-1 text-ui text-fg3">{m["library.subtitle"]()}</p>
      </header>

      <div className="flex min-h-0 flex-1 flex-col items-center justify-center py-16 text-center">
        <span className="grid size-10 place-items-center rounded-xl bg-tile text-fg3 shadow-[var(--shadow-hi)]">
          <Library className="size-4" strokeWidth={ICON_STROKE} />
        </span>
        <h2 className="mt-4 text-ui font-medium text-fg">{m["library.emptyTitle"]()}</h2>
        <p className="mt-1.5 max-w-[22rem] text-ui leading-6 text-fg3">
          {m["library.emptyDescription"]()}
        </p>
      </div>
    </div>
  );
}
