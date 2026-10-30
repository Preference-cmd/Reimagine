import { Activity, X } from "lucide-react";
import * as m from "$paraglide/messages";
import { ICON_STROKE } from "@/design/tokens";
import { useProjectStore } from "@/store/project";
import { useUIStore } from "@/store/uiStore";
import { ShellIconButton } from "./shellChrome";

/**
 * Floating session panel over the content area. Phase 1 ships the frame and
 * the workspace-activity tab only; the activity feed arrives in Phase 3.
 */
export function SessionPanel() {
  const open = useUIStore((state) => state.sessionPanelOpen);
  const setOpen = useUIStore((state) => state.setSessionPanelOpen);
  const projects = useProjectStore((state) => state.projects);
  const activeProjectId = useProjectStore((state) => state.activeProjectId);

  if (!open) return null;

  const title = projects.find((project) => project.id === activeProjectId)?.name ?? activeProjectId;

  return (
    <aside
      aria-label={m["shell.sessionPanel"]()}
      className="absolute top-4 right-4 bottom-4 z-20 flex w-[var(--spacing-panel)] flex-col overflow-hidden rounded-[18px] bg-surface shadow-[var(--shadow-bezel),var(--shadow-3),var(--shadow-hi)]"
    >
      <div className="flex items-center gap-0.5 px-4 pt-3 pb-2">
        <h2 className="min-w-0 flex-1 truncate text-sm font-semibold tracking-[-0.01em] text-fg">
          {title}
        </h2>
        <ShellIconButton label={m["shell.closePanel"]()} onClick={() => setOpen(false)}>
          <X className="size-4" strokeWidth={ICON_STROKE} />
        </ShellIconButton>
      </div>

      <div className="mx-4 mb-3 flex rounded-[11px] bg-app p-[3px] shadow-[inset_0_0_0_1px_var(--color-hair-soft)]">
        <div className="flex h-7 flex-1 items-center justify-center gap-1.5 rounded-sm bg-surface text-xs font-medium text-fg shadow-[0_0_0_1px_var(--color-hair),var(--shadow-1)]">
          <Activity className="size-3.5 text-fg2" strokeWidth={ICON_STROKE} />
          {m["shell.activity"]()}
        </div>
      </div>

      <div className="flex flex-1 flex-col items-center justify-center px-6 pb-12 text-center shadow-[inset_0_1px_0_var(--color-hair-soft)]">
        <span className="grid size-9 place-items-center rounded-lg bg-tile text-fg3 shadow-[var(--shadow-hi)]">
          <Activity className="size-4" strokeWidth={ICON_STROKE} />
        </span>
        <p className="mt-3 text-ui font-medium text-fg">{m["shell.noActivity"]()}</p>
        <p className="mt-1 max-w-56 text-xs leading-5 text-fg3">{m["shell.noActivityHint"]()}</p>
      </div>
    </aside>
  );
}
