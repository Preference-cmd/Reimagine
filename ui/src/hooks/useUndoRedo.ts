import { useEffect } from "react";
import { useWorkflowStore, useWorkflowTemporal } from "@/store/workflow";

export type BoardUndoTarget = {
  undo: () => void | Promise<unknown>;
  redo: () => void | Promise<unknown>;
};

let activeBoardUndoTarget: BoardUndoTarget | null = null;

export function registerBoardUndoTarget(target: BoardUndoTarget): void {
  activeBoardUndoTarget = target;
}

export function unregisterBoardUndoTarget(target: BoardUndoTarget): void {
  if (activeBoardUndoTarget === target) {
    activeBoardUndoTarget = null;
  }
}

/**
 * Bind Cmd/Ctrl+Z and Cmd/Ctrl+Shift+Z to either Board or Workflow undo / redo
 * depending on whether an active Board undo target is registered.
 * Mount once near the root.
 */
export function useUndoRedoShortcuts() {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (!mod) return;
      // Ignore when typing in an input/textarea/contenteditable so note / node
      // parameter editing doesn't hijack the shortcut.
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)
      ) {
        return;
      }
      const zKey = e.key === "z" || e.key === "Z";
      if (!zKey) return;
      e.preventDefault();

      if (activeBoardUndoTarget) {
        if (e.shiftKey) {
          void activeBoardUndoTarget.redo();
        } else {
          void activeBoardUndoTarget.undo();
        }
      } else {
        const t = useWorkflowStore.temporal.getState();
        if (e.shiftKey) t.redo();
        else t.undo();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);
}

/** Reactive hook for "can undo?" / "can redo?" booleans for workflows. */
export function useUndoRedoAvailability(): {
  canUndo: boolean;
  canRedo: boolean;
} {
  const canUndo = useWorkflowTemporal((s) => s.pastStates.length > 0);
  const canRedo = useWorkflowTemporal((s) => s.futureStates.length > 0);
  return { canUndo, canRedo };
}
