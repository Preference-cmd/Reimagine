import { create } from "zustand";
import {
  applyBoardCommands,
  getBoardSnapshot,
  previewBoardCommands,
  redoBoard,
  undoBoard,
} from "@/ipc";
import type { BoardCommandResultDto, BoardSnapshot } from "@/ipc/schemas";
import { useProjectStore } from "./project";

type BoardState = BoardSnapshot & {
  snapshotVersion: number;
  hydrate: (snapshot: BoardSnapshot) => void;
  hydrateIfNewer: (snapshot: BoardSnapshot) => boolean;
  load: (projectId: string) => Promise<boolean>;
  preview: (projectId: string, commandBatch: unknown) => Promise<BoardCommandResultDto>;
  apply: (projectId: string, commandBatch: unknown) => Promise<BoardCommandResultDto>;
  undo: (projectId: string) => Promise<BoardCommandResultDto | null>;
  redo: (projectId: string) => Promise<BoardCommandResultDto | null>;
};

const empty = (): BoardSnapshot & { snapshotVersion: number } => ({
  id: "",
  projectId: "default",
  version: 0,
  items: [],
  snapshotVersion: 0,
});

export const useBoardStore = create<BoardState>((set, get) => ({
  ...empty(),
  hydrate: (snapshot) =>
    set({
      ...snapshot,
      snapshotVersion: snapshot.version,
    }),
  // Applies a snapshot unless it is for an inactive project or older than
  // the version this store already holds for the same project.
  hydrateIfNewer: (snapshot) => {
    if (useProjectStore.getState().activeProjectId !== snapshot.projectId) {
      return false;
    }
    const current = get();
    if (current.projectId !== snapshot.projectId || snapshot.version >= current.version) {
      set({
        ...snapshot,
        snapshotVersion: snapshot.version,
      });
      return true;
    }
    return false;
  },
  load: async (projectId) => {
    const snapshot = await getBoardSnapshot(projectId);
    return get().hydrateIfNewer(snapshot);
  },
  preview: (projectId, commandBatch) => previewBoardCommands(projectId, commandBatch),
  apply: async (projectId, commandBatch) => {
    const result = await applyBoardCommands(projectId, commandBatch);
    if (get().projectId === projectId && result.version >= get().version) {
      set({ version: result.version });
    }
    return result;
  },
  undo: async (projectId) => {
    const result = await undoBoard(projectId);
    if (result && get().projectId === projectId && result.version >= get().version) {
      set({ version: result.version });
    }
    return result;
  },
  redo: async (projectId) => {
    const result = await redoBoard(projectId);
    if (result && get().projectId === projectId && result.version >= get().version) {
      set({ version: result.version });
    }
    return result;
  },
}));
