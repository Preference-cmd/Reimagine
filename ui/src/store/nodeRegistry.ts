import { create } from "zustand";
import { getNodeDefs } from "@/ipc";
import { NodeDefSchema, type NodeDef } from "@/ipc/schemas";

/**
 * Node catalog registry (F2-1).
 *
 * Fetches the backend node catalog once at startup via `getNodeDefs()` and
 * exposes it as a `Map<type_id, NodeDef>`. The canvas builds its React Flow
 * `nodeTypes` registry from this map; anything not covered by a hand-crafted
 * component falls back to GenericNode.
 *
 * Failure is non-fatal: the store lands in `error` with an empty map and the
 * canvas renders every node through GenericNode with whatever data it has.
 */

export type NodeRegistryStatus = "idle" | "loading" | "ready" | "error";

type NodeRegistryState = {
  status: NodeRegistryStatus;
  /** type_id → node def (stable map identity until the next load). */
  defs: Map<string, NodeDef>;
  /** Same defs as a list (handy for iteration/indexing). */
  defList: NodeDef[];
  error: string | null;
  load: () => Promise<void>;
};

export const useNodeRegistryStore = create<NodeRegistryState>()((set, get) => ({
  status: "idle",
  defs: new Map(),
  defList: [],
  error: null,

  load: async () => {
    if (get().status === "loading") return;
    set({ status: "loading", error: null });
    try {
      const raw = await getNodeDefs();
      // Per-def safe parsing: a malformed entry is dropped, never fatal.
      const defs: NodeDef[] = [];
      for (const entry of raw) {
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
    } catch (err) {
      // Empty catalog → every node renders via GenericNode fallback.
      console.warn(
        "[nodeRegistry] catalog fetch failed; falling back to GenericNode",
        err,
      );
      set({ status: "error", error: String(err) });
    }
  },
}));

/** Selector: node def for a type id (undefined when the catalog lacks it). */
export const selectNodeDef = (
  defs: Map<string, NodeDef>,
  type: string | null | undefined,
): NodeDef | undefined => (type ? defs.get(type) : undefined);
