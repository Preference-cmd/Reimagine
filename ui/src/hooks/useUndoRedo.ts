import { useEffect } from "react";
import { showsSessionPanel } from "@/components/layout/shellChrome";
import { runBoardHistory } from "@/lib/boardHistory";

/**
 * Bind Cmd/Ctrl+Z and Cmd/Ctrl+Shift+Z to the active project's board undo /
 * redo. The app root mounts this once, so the shortcut keeps working on
 * surfaces that do not render the canvas; both keys go through the same
 * history guard as the activity panel's Undo and Redo buttons. Outside the
 * canvas and library modes the key is left to the browser.
 */
export function useUndoRedoShortcuts() {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.repeat || e.altKey) return;
      const mod = e.metaKey || e.ctrlKey;
      if (!mod) return;
      const zKey = e.key === "z" || e.key === "Z";
      if (!zKey) return;
      // Check the route before claiming the key, so /settings, /models and
      // collection details keep their browser undo.
      if (!showsSessionPanel(window.location.pathname)) return;
      // Ignore when typing in an input/textarea/contenteditable so note
      // editing doesn't hijack the shortcut.
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)
      ) {
        return;
      }
      e.preventDefault();

      void runBoardHistory(e.shiftKey ? "redo" : "undo");
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);
}
