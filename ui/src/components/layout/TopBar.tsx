import {
  Play,
  Undo2,
  Redo2,
  ChevronDown,
  StickyNote,
  Cpu,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useWorkflowStore } from "@/store/workflow";
import { useUndoRedoAvailability } from "@/hooks/useUndoRedo";

const NAV_TABS = ["Workflow", "Edit", "View"] as const;

export function TopBar() {
  const { canUndo, canRedo } = useUndoRedoAvailability();
  const undo = () => useWorkflowStore.temporal.getState().undo();
  const redo = () => useWorkflowStore.temporal.getState().redo();

  return (
    <header className="absolute top-0 left-0 right-0 z-50 flex h-14 items-center gap-4 border-b border-white/5 px-4 backdrop-blur-md">
      <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-gray-700 to-gray-900 shadow-lg">
        <Cpu className="h-4 w-4 text-zinc-400" />
      </div>

      <nav className="flex items-center gap-1 rounded-xl border border-white/5 bg-zinc-900/50 p-1">
        {NAV_TABS.map((tab, i) => (
          <button
            key={tab}
            className={
              i === 0
                ? "rounded-lg bg-zinc-800 px-3 py-1.5 text-sm font-medium text-white"
                : "rounded-lg px-3 py-1.5 text-sm font-medium text-zinc-400 transition-colors hover:text-white"
            }
          >
            {tab}
          </button>
        ))}
        <div className="mx-1 h-4 w-px bg-white/10" />
        <Button
          size="icon-sm"
          variant="ghost"
          className="text-zinc-400 hover:text-white"
        >
          <Play />
        </Button>
      </nav>

      <div className="flex gap-1">
        <Button
          size="icon-sm"
          variant="ghost"
          onClick={undo}
          disabled={!canUndo}
          className="text-zinc-500 hover:text-white disabled:opacity-30"
          title="Undo (Cmd/Ctrl+Z)"
        >
          <Undo2 />
        </Button>
        <Button
          size="icon-sm"
          variant="ghost"
          onClick={redo}
          disabled={!canRedo}
          className="text-zinc-500 hover:text-white disabled:opacity-30"
          title="Redo (Cmd/Ctrl+Shift+Z)"
        >
          <Redo2 />
        </Button>
      </div>

      <div className="flex items-center gap-2 rounded-xl border border-white/5 bg-zinc-900/50 px-3 py-1.5">
        <StickyNote className="h-4 w-4 text-zinc-500" />
        <span className="text-sm font-medium text-zinc-200">
          Untitled workflow
        </span>
        <span className="h-2 w-2 rounded-full bg-zinc-700" />
        <ChevronDown className="h-3.5 w-3.5 text-zinc-500" />
      </div>
    </header>
  );
}