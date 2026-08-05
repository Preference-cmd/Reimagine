import { useEffect, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { useWorkflowStore } from "@/store/workflow";
import { useUIStore } from "@/store/uiStore";

/**
 * Node rename dialog (F3-1): driven by the UI store's `renameTarget`,
 * opened from the node context menu's "Rename…" item.
 */
export function RenameNodeDialog() {
  const target = useUIStore((s) => s.renameTarget);
  const finishRename = useUIStore((s) => s.finishRename);
  const renameNode = useWorkflowStore((s) => s.renameNode);
  const [value, setValue] = useState("");

  useEffect(() => {
    if (target) setValue(target.title);
  }, [target]);

  const commit = () => {
    const title = value.trim();
    if (target && title && title !== target.title) {
      renameNode(target.id, title);
    }
    finishRename();
  };

  return (
    <Dialog open={target != null} onOpenChange={(open) => !open && finishRename()}>
      <DialogContent className="w-80">
        <DialogHeader>
          <DialogTitle>Rename node</DialogTitle>
        </DialogHeader>
        <input
          autoFocus
          className="w-full rounded-md border border-control-border bg-surface-container-high px-2.5 py-1.5 text-body-sm text-on-surface placeholder:text-on-surface-variant focus:outline-none focus:ring-1 focus:ring-control-active"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") commit();
          }}
          placeholder="Node name"
        />
        <DialogFooter>
          <Button variant="outline" onClick={finishRename}>
            Cancel
          </Button>
          <Button onClick={commit}>Rename</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
