import { create } from "zustand";
import { applyBoardCommands, getBoardSnapshot, previewBoardCommands, redoBoard, undoBoard } from "@/ipc";
import type { BoardCommandResult, BoardSnapshot } from "@/ipc/schemas";

type BoardState = BoardSnapshot & {
  hydrate: (snapshot: BoardSnapshot) => void;
  load: (projectId: string) => Promise<void>;
  acceptVersion: (projectId: string, version: number) => boolean;
  preview: (projectId: string, commandBatch: unknown) => Promise<BoardCommandResult>;
  apply: (projectId: string, commandBatch: unknown) => Promise<BoardCommandResult>;
  undo: (projectId: string) => Promise<BoardCommandResult | null>;
  redo: (projectId: string) => Promise<BoardCommandResult | null>;
};

const empty = (): BoardSnapshot => ({ id: "", projectId: "default", version: 0, items: [] });

export const useBoardStore = create<BoardState>((set, get) => ({
  ...empty(),
  hydrate: (snapshot) => set(snapshot),
  load: async (projectId) => { set(await getBoardSnapshot(projectId)); },
  acceptVersion: (projectId, version) => {
    const state = get();
    if (state.projectId !== projectId || version < state.version) return false;
    set({ projectId, version });
    return true;
  },
  preview: (projectId, commandBatch) => previewBoardCommands(projectId, commandBatch),
  apply: async (projectId, commandBatch) => {
    const result = await applyBoardCommands(projectId, commandBatch);
    if (result.boardVersion >= get().version) set({ projectId, version: result.boardVersion });
    return result;
  },
  undo: async (projectId) => {
    const result = await undoBoard(projectId);
    if (result && result.boardVersion >= get().version) set({ projectId, version: result.boardVersion });
    return result;
  },
  redo: async (projectId) => {
    const result = await redoBoard(projectId);
    if (result && result.boardVersion >= get().version) set({ projectId, version: result.boardVersion });
    return result;
  },
}));