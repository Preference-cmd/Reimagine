import { useUIStore } from "@/store/uiStore";
import { useSettingsStore } from "@/store/settingsStore";
import { cn } from "@/lib/utils";
import * as m from "$paraglide/messages";

export type ThemeMode = "light" | "dark";

type SettingItem = {
  id: string;
  labelKey: string;
  descriptionKey?: string;
  type: "toggle" | "select" | "segment";
  options?: string[];
};

type SettingsGroup = {
  labelKey: string;
  items: SettingItem[];
};

const SETTINGS_CONTENT: Record<string, SettingsGroup[]> = {
  general: [
    {
      labelKey: "settings.general",
      items: [
        { id: "auto-save", labelKey: "settings.auto-save", descriptionKey: "settings.auto-save-desc", type: "toggle" },
        { id: "restore-session", labelKey: "settings.restore-session", descriptionKey: "settings.restore-session-desc", type: "toggle" },
        { id: "check-updates", labelKey: "settings.check-updates", descriptionKey: "settings.check-updates-desc", type: "toggle" },
      ],
    },
  ],
  appearance: [
    {
      labelKey: "settings.appearance",
      items: [
        { id: "color-mode", labelKey: "settings.color-mode", descriptionKey: "settings.color-mode-desc", type: "segment", options: ["settings.light", "settings.dark"] },
      ],
    },
    {
      labelKey: "settings.appearance",
      items: [
        { id: "grid-style", labelKey: "settings.grid-style", descriptionKey: "settings.grid-style-desc", type: "segment", options: ["settings.dots", "settings.lines", "settings.none"] },
        { id: "minimap", labelKey: "settings.minimap", descriptionKey: "settings.minimap-desc", type: "toggle" },
      ],
    },
  ],
  shortcuts: [
    {
      labelKey: "settings.shortcuts",
      items: [
        { id: "cmd-palette", labelKey: "settings.cmd-palette", type: "segment", options: ["⌘P", "⌘K"] },
        { id: "save", labelKey: "common.save", type: "segment", options: ["⌘S"] },
        { id: "undo", labelKey: "settings.undo", type: "segment", options: ["⌘Z"] },
        { id: "redo", labelKey: "settings.redo", type: "segment", options: ["⌘⇧Z"] },
      ],
    },
  ],
  runtime: [
    {
      labelKey: "settings.runtime",
      items: [
        { id: "backend", labelKey: "settings.default-backend", descriptionKey: "settings.default-backend-desc", type: "select", options: ["settings.burn", "settings.candle"] },
        { id: "device", labelKey: "settings.device-selection", descriptionKey: "settings.device-selection-desc", type: "select", options: ["settings.auto-detect", "settings.cpu", "settings.gpu-0", "settings.gpu-1"] },
        { id: "memory-budget", labelKey: "settings.vram-budget", descriptionKey: "settings.vram-budget-desc", type: "select", options: ["settings.2gb", "settings.4gb", "settings.8gb", "settings.16gb"] },
      ],
    },
  ],
  workspace: [
    {
      labelKey: "settings.workspace",
      items: [
        { id: "project-dir", labelKey: "settings.project-directory", descriptionKey: "settings.project-directory-desc", type: "select", options: ["~/Reimagine", "~/Documents/Reimagine"] },
        { id: "autosave-interval", labelKey: "settings.autosave-interval", type: "select", options: ["settings.5-seconds", "settings.10-seconds", "settings.30-seconds", "settings.1-minute"] },
      ],
    },
  ],
  models: [
    {
      labelKey: "settings.modelManagement",
      items: [
        { id: "download-dir", labelKey: "settings.download-directory", descriptionKey: "settings.download-directory-desc", type: "select", options: ["~/.cache/reimagine/models", "~/Reimagine/models"] },
        { id: "auto-convert", labelKey: "settings.auto-convert", descriptionKey: "settings.auto-convert-desc", type: "toggle" },
      ],
    },
  ],
};

const SECTION_LABEL_KEYS: Record<string, string> = {
  general: "settings.general",
  appearance: "settings.appearance",
  shortcuts: "settings.shortcuts",
  runtime: "settings.runtime",
  workspace: "settings.workspace",
  models: "settings.modelManagement",
};

type SettingsViewProps = {
  themeMode: ThemeMode;
  onThemeModeChange: (mode: ThemeMode) => void;
};

/**
 * SettingsView -- content-only settings page.
 * The sidebar handles navigation; this component renders the form.
 */
export function SettingsView({ themeMode, onThemeModeChange }: SettingsViewProps) {
  const settingsNavId = useUIStore((s) => s.settingsNavId);
  const activeSection = settingsNavId ?? "general";
  const groups = SETTINGS_CONTENT[activeSection] ?? [];
  const sectionLabelKey = SECTION_LABEL_KEYS[activeSection] ?? "settings.general";

  // Resolve message keys to actual translated strings
  const resolveLabel = (key: string): string => {
    const msg = (m as unknown as Record<string, () => string>)[key];
    return msg ? msg() : key;
  };

  return (
    <div className="min-h-0 flex-1 overflow-y-auto p-8">
      <h1 className="mb-8 text-display-sm font-semibold text-on-surface">
        {resolveLabel(sectionLabelKey)}
      </h1>

      {groups.map((group, gi) => (
        <section key={`${group.labelKey}-${gi}`} className="mb-8">
          <h2 className="mb-3 text-body-sm font-semibold text-on-surface">
            {resolveLabel(group.labelKey)}
          </h2>
          <div className="divide-y divide-outline rounded-xl border border-outline bg-surface">
            {group.items.map((item) => (
              <SettingRow key={item.id} item={item} themeMode={themeMode} onThemeModeChange={onThemeModeChange} resolveLabel={resolveLabel} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

function SettingRow({
  item,
  themeMode,
  onThemeModeChange,
  resolveLabel,
}: {
  item: SettingItem;
  themeMode: ThemeMode;
  onThemeModeChange: (mode: ThemeMode) => void;
  resolveLabel: (key: string) => string;
}) {
  const settings = useSettingsStore();
  const segmentIndex = useSettingsStore((s) => {
    if (item.id === "grid-style") {
      return s.gridStyle === "dots" ? 0 : s.gridStyle === "lines" ? 1 : 2;
    }
    if (item.id === "cmd-palette") return s.cmdPaletteKey === "⌘P" ? 0 : 1;
    return 0;
  });

  const isColorMode = item.id === "color-mode";
  const isGridStyle = item.id === "grid-style";
  const isCmdPalette = item.id === "cmd-palette";
  const currentSegment = isColorMode
    ? themeMode === "dark" ? 1 : 0
    : segmentIndex;

  const toggleValue = (() => {
    switch (item.id) {
      case "auto-save": return settings.autoSave;
      case "restore-session": return settings.restoreSession;
      case "check-updates": return settings.checkUpdates;
      case "minimap": return settings.minimap;
      case "auto-convert": return settings.autoConvert;
      default: return false;
    }
  })();

  const selectValue = (() => {
    switch (item.id) {
      case "backend": return settings.backend === "Burn" ? resolveLabel("settings.burn") : resolveLabel("settings.candle");
      case "device": return settings.device === "auto" ? resolveLabel("settings.auto-detect") : settings.device === "cpu" ? resolveLabel("settings.cpu") : settings.device === "gpu0" ? resolveLabel("settings.gpu-0") : resolveLabel("settings.gpu-1");
      case "memory-budget": return settings.memoryBudget === "2GB" ? resolveLabel("settings.2gb") : settings.memoryBudget === "4GB" ? resolveLabel("settings.4gb") : settings.memoryBudget === "8GB" ? resolveLabel("settings.8gb") : resolveLabel("settings.16gb");
      case "project-dir": return settings.projectDir;
      case "autosave-interval": return settings.autosaveInterval === "5s" ? resolveLabel("settings.5-seconds") : settings.autosaveInterval === "10s" ? resolveLabel("settings.10-seconds") : settings.autosaveInterval === "30s" ? resolveLabel("settings.30-seconds") : resolveLabel("settings.1-minute");
      case "download-dir": return settings.downloadDir;
      default: return "";
    }
  })();

  const handleToggle = () => {
    switch (item.id) {
      case "auto-save": settings.setAutoSave(!toggleValue); break;
      case "restore-session": settings.setRestoreSession(!toggleValue); break;
      case "check-updates": settings.setCheckUpdates(!toggleValue); break;
      case "minimap": settings.setMinimap(!toggleValue); break;
      case "auto-convert": settings.setAutoConvert(!toggleValue); break;
    }
  };

  const handleSelect = (value: string) => {
    switch (item.id) {
      case "backend": settings.setBackend(value.includes("Candle") ? "Candle (deprecated)" : "Burn"); break;
      case "device": {
        const v = value.includes("Auto") ? "auto" : value.includes("CPU") ? "cpu" : value.includes("0") ? "gpu0" : "gpu1";
        settings.setDevice(v as "auto" | "cpu" | "gpu0" | "gpu1");
        break;
      }
      case "memory-budget": {
        const v = value.includes("2") ? "2GB" : value.includes("4") ? "4GB" : value.includes("8") ? "8GB" : "16GB";
        settings.setMemoryBudget(v as "2GB" | "4GB" | "8GB" | "16GB");
        break;
      }
      case "project-dir": settings.setProjectDir(value); break;
      case "autosave-interval": {
        const v = value.includes("5") ? "5s" : value.includes("10") ? "10s" : value.includes("30") ? "30s" : "60s";
        settings.setAutosaveInterval(v as "5s" | "10s" | "30s" | "60s");
        break;
      }
      case "download-dir": settings.setDownloadDir(value); break;
    }
  };

  const handleSegment = (i: number) => {
    if (isColorMode) {
      onThemeModeChange(i === 0 ? "light" : "dark");
    } else if (isGridStyle) {
      settings.setGridStyle(i === 0 ? "dots" : i === 1 ? "lines" : "none");
    } else if (isCmdPalette) {
      settings.setCmdPaletteKey(i === 0 ? "⌘P" : "⌘K");
    }
  };

  return (
    <div className="flex items-center justify-between gap-4 px-4 py-3">
      <div className="min-w-0 flex-1">
        <div className="text-body-sm font-medium text-on-surface">
          {resolveLabel(item.labelKey)}
        </div>
        {item.descriptionKey && (
          <div className="mt-0.5 text-caption text-on-surface-variant">
            {resolveLabel(item.descriptionKey)}
          </div>
        )}
      </div>

      <div className="shrink-0">
        {item.type === "toggle" && (
          <button
            type="button"
            role="switch"
            aria-checked={toggleValue ? "true" : "false"}
            onClick={handleToggle}
            className={cn(
              "relative h-6 w-11 shrink-0 rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-secondary/30",
              toggleValue ? "bg-secondary" : "bg-control-border",
            )}
          >
            <span
              className={cn(
                "absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-white shadow-sm transition-transform",
                toggleValue && "translate-x-5",
              )}
            />
          </button>
        )}

        {item.type === "select" && (
          <select
            className="h-8 rounded-lg border border-outline bg-surface px-3 text-body-sm text-on-surface outline-none focus-visible:border-secondary/30 focus-visible:ring-2 focus-visible:ring-secondary/10"
            value={selectValue}
            onChange={(e) => handleSelect(e.target.value)}
          >
            {item.options?.map((opt) => (
              <option key={opt} value={resolveLabel(opt)}>
                {resolveLabel(opt)}
              </option>
            ))}
          </select>
        )}

        {item.type === "segment" && item.options && (
          <div className="flex rounded-lg border border-outline/50 bg-surface-container-low p-1">
            {item.options.map((opt, i) => (
              <button
                key={opt}
                type="button"
                onClick={() => handleSegment(i)}
                className={cn(
                  "flex-1 rounded-md px-3 py-1.5 text-caption font-medium transition-all",
                  currentSegment === i
                    ? "bg-surface text-on-surface shadow-sm"
                    : "text-on-surface-variant hover:text-on-surface hover:bg-control-hover",
                )}
              >
                {resolveLabel(opt)}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
