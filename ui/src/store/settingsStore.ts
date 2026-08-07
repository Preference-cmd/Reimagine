import { create } from "zustand";

/**
 * Settings persistence store — persists user preferences to localStorage.
 * Theme mode is handled separately in AppShell.
 */

const STORAGE_KEY = "reimagine.settings";

type SettingsState = {
  // General
  autoSave: boolean;
  restoreSession: boolean;
  checkUpdates: boolean;

  // Appearance
  gridStyle: "dots" | "lines" | "none";
  minimap: boolean;

  // Shortcuts
  cmdPaletteKey: "⌘P" | "⌘K";

  // Runtime
  backend: "Burn" | "Candle (deprecated)";
  device: "auto" | "cpu" | "gpu0" | "gpu1";
  memoryBudget: "2GB" | "4GB" | "8GB" | "16GB";

  // Workspace
  projectDir: string;
  autosaveInterval: "5s" | "10s" | "30s" | "60s";

  // Models
  downloadDir: string;
  autoConvert: boolean;

  // Actions
  setAutoSave: (v: boolean) => void;
  setRestoreSession: (v: boolean) => void;
  setCheckUpdates: (v: boolean) => void;
  setGridStyle: (v: "dots" | "lines" | "none") => void;
  setMinimap: (v: boolean) => void;
  setCmdPaletteKey: (v: "⌘P" | "⌘K") => void;
  setBackend: (v: "Burn" | "Candle (deprecated)") => void;
  setDevice: (v: "auto" | "cpu" | "gpu0" | "gpu1") => void;
  setMemoryBudget: (v: "2GB" | "4GB" | "8GB" | "16GB") => void;
  setProjectDir: (v: string) => void;
  setAutosaveInterval: (v: "5s" | "10s" | "30s" | "60s") => void;
  setDownloadDir: (v: string) => void;
  setAutoConvert: (v: boolean) => void;
};

function loadSettings(): Partial<SettingsState> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function persistSettings(state: SettingsState) {
  const {
    setAutoSave: _setAutoSave,
    setRestoreSession: _setRestoreSession,
    setCheckUpdates: _setCheckUpdates,
    setGridStyle: _setGridStyle,
    setMinimap: _setMinimap,
    setCmdPaletteKey: _setCmdPaletteKey,
    setBackend: _setBackend,
    setDevice: _setDevice,
    setMemoryBudget: _setMemoryBudget,
    setProjectDir: _setProjectDir,
    setAutosaveInterval: _setAutosaveInterval,
    setDownloadDir: _setDownloadDir,
    setAutoConvert: _setAutoConvert,
    ...settings
  } = state;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}

const DEFAULTS: Omit<
  SettingsState,
  | "setAutoSave"
  | "setRestoreSession"
  | "setCheckUpdates"
  | "setGridStyle"
  | "setMinimap"
  | "setCmdPaletteKey"
  | "setBackend"
  | "setDevice"
  | "setMemoryBudget"
  | "setProjectDir"
  | "setAutosaveInterval"
  | "setDownloadDir"
  | "setAutoConvert"
> = {
  autoSave: true,
  restoreSession: true,
  checkUpdates: true,
  gridStyle: "dots",
  minimap: true,
  cmdPaletteKey: "⌘P",
  backend: "Burn",
  device: "auto",
  memoryBudget: "4GB",
  projectDir: "~/Reimagine",
  autosaveInterval: "5s",
  downloadDir: "~/.cache/reimagine/models",
  autoConvert: true,
};

export const useSettingsStore = create<SettingsState>()((set, get) => {
  const stored = loadSettings();
  const initial = { ...DEFAULTS, ...stored };

  const persist = () => persistSettings(get());

  return {
    ...initial,

    setAutoSave: (v) => {
      set({ autoSave: v });
      persist();
    },
    setRestoreSession: (v) => {
      set({ restoreSession: v });
      persist();
    },
    setCheckUpdates: (v) => {
      set({ checkUpdates: v });
      persist();
    },
    setGridStyle: (v) => {
      set({ gridStyle: v });
      persist();
    },
    setMinimap: (v) => {
      set({ minimap: v });
      persist();
    },
    setCmdPaletteKey: (v) => {
      set({ cmdPaletteKey: v });
      persist();
    },
    setBackend: (v) => {
      set({ backend: v });
      persist();
    },
    setDevice: (v) => {
      set({ device: v });
      persist();
    },
    setMemoryBudget: (v) => {
      set({ memoryBudget: v });
      persist();
    },
    setProjectDir: (v) => {
      set({ projectDir: v });
      persist();
    },
    setAutosaveInterval: (v) => {
      set({ autosaveInterval: v });
      persist();
    },
    setDownloadDir: (v) => {
      set({ downloadDir: v });
      persist();
    },
    setAutoConvert: (v) => {
      set({ autoConvert: v });
      persist();
    },
  };
});
