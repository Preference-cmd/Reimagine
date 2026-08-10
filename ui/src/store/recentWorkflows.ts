import { create } from "zustand";
import { persist } from "zustand/middleware";

type RecentWorkflowEntry = {
  id: string;
  name: string;
  lastOpened: number;
};

type RecentWorkflowsState = {
  entries: RecentWorkflowEntry[];
  addRecent: (id: string, name: string) => void;
  removeRecent: (id: string) => void;
  clearRecent: () => void;
  updateNames: (updates: Array<{ id: string; name: string }>) => void;
};

const MAX_RECENT = 10;
const STORAGE_KEY = "reimagine:recent-workflows";

export const useRecentWorkflowsStore = create<RecentWorkflowsState>()(
  persist(
    (set, get) => ({
      entries: [],

      addRecent: (id: string, name: string) => {
        const entries = get().entries.filter((e) => e.id !== id);
        entries.unshift({ id, name, lastOpened: Date.now() });
        set({ entries: entries.slice(0, MAX_RECENT) });
      },

      removeRecent: (id: string) => {
        set({ entries: get().entries.filter((e) => e.id !== id) });
      },

      clearRecent: () => set({ entries: [] }),

      updateNames: (updates: Array<{ id: string; name: string }>) => {
        const nameMap = new Map(updates.map((u) => [u.id, u.name]));
        set({
          entries: get().entries.map((e) => {
            const newName = nameMap.get(e.id);
            return newName ? { ...e, name: newName } : e;
          }),
        });
      },
    }),
    {
      name: STORAGE_KEY,
      partialize: (state) => ({ entries: state.entries }),
    },
  ),
);
