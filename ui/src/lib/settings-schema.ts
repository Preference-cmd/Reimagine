import { z } from "zod";

export const settingsSchema = z.object({
  // General
  autoSave: z.boolean(),
  restoreSession: z.boolean(),
  checkUpdates: z.boolean(),

  // Appearance
  gridStyle: z.enum(["dots", "lines", "none"]),
  minimap: z.boolean(),

  // Shortcuts
  cmdPaletteKey: z.enum(["⌘P", "⌘K"]),

  // Runtime
  backend: z.enum(["Burn", "Candle (deprecated)"]),
  device: z.enum(["auto", "cpu", "gpu0", "gpu1"]),
  memoryBudget: z.enum(["2GB", "4GB", "8GB", "16GB"]),

  // Workspace
  projectDir: z.string().min(1),
  autosaveInterval: z.enum(["5s", "10s", "30s", "60s"]),

  // Models
  downloadDir: z.string().min(1),
  autoConvert: z.boolean(),
});

export type SettingsFormData = z.infer<typeof settingsSchema>;
