import { create } from "zustand";
import { NodeDefSchema, type NodeDef } from "@/ipc/schemas";

/**
 * Node catalog registry (F2-1).
 *
 * Populated at startup from the `useNodeDefs()` query (see hooks/queries.ts).
 * The `hydrate()` action transforms the raw IPC array into the Map structure
 * that canvas components consume. Failure is non-fatal: the store lands in
 * `error` with an empty map and the canvas renders every node through
 * GenericNode with whatever data it has.
 */

export type NodeRegistryStatus = "idle" | "loading" | "ready" | "error";

type NodeRegistryState = {
  status: NodeRegistryStatus;
  /** type_id → node def (stable map identity until the next load). */
  defs: Map<string, NodeDef>;
  /** Same defs as a list (handy for iteration/indexing). */
  defList: NodeDef[];
  error: string | null;
  /** Populate the store from pre-fetched data (called by __root.tsx). */
  hydrate: (defs: NodeDef[], error?: string) => void;
};

export const useNodeRegistryStore = create<NodeRegistryState>()((set) => ({
  status: "idle",
  defs: new Map(),
  defList: [],
  error: null,

  hydrate: (rawDefs: NodeDef[], error?: string) => {
    if (error) {
      console.warn("[nodeRegistry] catalog fetch failed; falling back to GenericNode", error);
      set({ status: "error", error });
      return;
    }
    // Per-def safe parsing: a malformed entry is dropped, never fatal.
    const defs: NodeDef[] = [];
    for (const entry of rawDefs) {
      const parsed = NodeDefSchema.safeParse(entry);
      if (parsed.success) {
        defs.push(parsed.data);
      } else {
        console.warn(
          "[nodeRegistry] skipping malformed node def",
          entry?.type ?? "<unknown>",
          parsed.error,
        );
      }
    }
    set({
      status: "ready",
      defList: defs,
      defs: new Map(defs.map((def) => [def.type, def])),
    });
  },
}));

/** Selector: node def for a type id (undefined when the catalog lacks it). */
export const selectNodeDef = (
  defs: Map<string, NodeDef>,
  type: string | null | undefined,
): NodeDef | undefined => (type ? defs.get(type) : undefined);
