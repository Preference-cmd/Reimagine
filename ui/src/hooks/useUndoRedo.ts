import { useEffect } from "react";
import { useWorkflowStore, useWorkflowTemporal } from "@/store/workflow";

/**
 * Bind Cmd/Ctrl+Z and Cmd/Ctrl+Shift+Z to the workflow's undo / redo.
 * Mount once near the root.
 */
export function useUndoRedoShortcuts() {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (!mod) return;
      // Ignore when typing in an input/textarea/contenteditable so node
      // parameter editing doesn't hijack the shortcut.
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable)
      ) {
        return;
      }
      const zKey = e.key === "z" || e.key === "Z";
      if (!zKey) return;
      e.preventDefault();
      const t = useWorkflowStore.temporal.getState();
      if (e.shiftKey) t.redo();
      else t.undo();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);
}

/** Reactive hook for "can undo?" / "can redo?" booleans. */
export function useUndoRedoAvailability(): {
  canUndo: boolean;
  canRedo: boolean;
} {
  const canUndo = useWorkflowTemporal((s) => s.pastStates.length > 0);
  const canRedo = useWorkflowTemporal((s) => s.futureStates.length > 0);
  return { canUndo, canRedo };
}