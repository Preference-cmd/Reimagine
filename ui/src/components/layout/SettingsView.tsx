import { useUIStore } from "@/store/uiStore";
import { useSettingsStore } from "@/store/settingsStore";
import { cn } from "@/lib/utils";
import * as m from "$paraglide/messages";
import { setLocale, getLocale } from "$paraglide/runtime";
import { useForm, type UseFormReturn } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { settingsSchema, type SettingsFormData } from "@/lib/settings-schema";
import { useEffect, useRef, useState } from "react";

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
        {
          id: "language",
          labelKey: "settings.language",
          type: "segment",
          options: ["English", "中文"],
        },
        {
          id: "auto-save",
          labelKey: "settings.auto-save",
          descriptionKey: "settings.auto-save-desc",
          type: "toggle",
        },
        {
          id: "restore-session",
          labelKey: "settings.restore-session",
          descriptionKey: "settings.restore-session-desc",
          type: "toggle",
        },
        {
          id: "check-updates",
          labelKey: "settings.check-updates",
          descriptionKey: "settings.check-updates-desc",
          type: "toggle",
        },
      ],
    },
  ],
  appearance: [
    {
      labelKey: "settings.appearance",
      items: [
        {
          id: "color-mode",
          labelKey: "settings.color-mode",
          descriptionKey: "settings.color-mode-desc",
          type: "segment",
          options: ["settings.light", "settings.dark"],
        },
      ],
    },
    {
      labelKey: "settings.appearance",
      items: [
        {
          id: "grid-style",
          labelKey: "settings.grid-style",
          descriptionKey: "settings.grid-style-desc",
          type: "segment",
          options: ["settings.dots", "settings.lines", "settings.none"],
        },
        {
          id: "minimap",
          labelKey: "settings.minimap",
          descriptionKey: "settings.minimap-desc",
          type: "toggle",
        },
      ],
    },
  ],
  shortcuts: [
    {
      labelKey: "settings.shortcuts",
      items: [
        {
          id: "cmd-palette",
          labelKey: "settings.cmd-palette",
          type: "segment",
          options: ["⌘P", "⌘K"],
        },
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
        {
          id: "backend",
          labelKey: "settings.default-backend",
          descriptionKey: "settings.default-backend-desc",
          type: "select",
          options: ["settings.burn", "settings.candle"],
        },
        {
          id: "device",
          labelKey: "settings.device-selection",
          descriptionKey: "settings.device-selection-desc",
          type: "select",
          options: ["settings.auto-detect", "settings.cpu", "settings.gpu-0", "settings.gpu-1"],
        },
        {
          id: "memory-budget",
          labelKey: "settings.vram-budget",
          descriptionKey: "settings.vram-budget-desc",
          type: "select",
          options: ["settings.2gb", "settings.4gb", "settings.8gb", "settings.16gb"],
        },
      ],
    },
  ],
  workspace: [
    {
      labelKey: "settings.workspace",
      items: [
        {
          id: "project-dir",
          labelKey: "settings.project-directory",
          descriptionKey: "settings.project-directory-desc",
          type: "select",
          options: ["~/Reimagine", "~/Documents/Reimagine"],
        },
        {
          id: "autosave-interval",
          labelKey: "settings.autosave-interval",
          type: "select",
          options: [
            "settings.5-seconds",
            "settings.10-seconds",
            "settings.30-seconds",
            "settings.1-minute",
          ],
        },
      ],
    },
  ],
  models: [
    {
      labelKey: "settings.modelManagement",
      items: [
        {
          id: "download-dir",
          labelKey: "settings.download-directory",
          descriptionKey: "settings.download-directory-desc",
          type: "select",
          options: ["~/.cache/reimagine/models", "~/Reimagine/models"],
        },
        {
          id: "auto-convert",
          labelKey: "settings.auto-convert",
          descriptionKey: "settings.auto-convert-desc",
          type: "toggle",
        },
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
 * SettingsView — Codex-style content-only settings page.
 * Light gray background, large title, card sections with group headers.
 */
export function SettingsView({ themeMode, onThemeModeChange }: SettingsViewProps) {
  const settingsNavId = useUIStore((s) => s.settingsNavId);
  const activeSection = settingsNavId ?? "general";
  const groups = SETTINGS_CONTENT[activeSection] ?? [];
  const sectionLabelKey = SECTION_LABEL_KEYS[activeSection] ?? "settings.general";

  const settings = useSettingsStore();

  const form = useForm<SettingsFormData>({
    resolver: zodResolver(settingsSchema),
    defaultValues: {
      autoSave: settings.autoSave,
      restoreSession: settings.restoreSession,
      checkUpdates: settings.checkUpdates,
      gridStyle: settings.gridStyle,
      minimap: settings.minimap,
      cmdPaletteKey: settings.cmdPaletteKey,
      backend: settings.backend,
      device: settings.device,
      memoryBudget: settings.memoryBudget,
      projectDir: settings.projectDir,
      autosaveInterval: settings.autosaveInterval,
      downloadDir: settings.downloadDir,
      autoConvert: settings.autoConvert,
    },
  });

  useEffect(() => {
    const subscription = form.watch((value) => {
      if (!value) return;
      if (value.autoSave !== undefined) settings.setAutoSave(value.autoSave);
      if (value.restoreSession !== undefined) settings.setRestoreSession(value.restoreSession);
      if (value.checkUpdates !== undefined) settings.setCheckUpdates(value.checkUpdates);
      if (value.gridStyle !== undefined) settings.setGridStyle(value.gridStyle);
      if (value.minimap !== undefined) settings.setMinimap(value.minimap);
      if (value.cmdPaletteKey !== undefined) settings.setCmdPaletteKey(value.cmdPaletteKey);
      if (value.backend !== undefined) settings.setBackend(value.backend);
      if (value.device !== undefined) settings.setDevice(value.device);
      if (value.memoryBudget !== undefined) settings.setMemoryBudget(value.memoryBudget);
      if (value.projectDir !== undefined) settings.setProjectDir(value.projectDir);
      if (value.autosaveInterval !== undefined)
        settings.setAutosaveInterval(value.autosaveInterval);
      if (value.downloadDir !== undefined) settings.setDownloadDir(value.downloadDir);
      if (value.autoConvert !== undefined) settings.setAutoConvert(value.autoConvert);
    });
    return () => subscription.unsubscribe();
  }, [form, settings]);

  const resolveLabel = (key: string): string => {
    const msg = (m as unknown as Record<string, () => string>)[key];
    return msg ? msg() : key;
  };

  return (
    <div className="min-h-0 flex-1 overflow-y-auto bg-background">
      <div className="max-w-[750px] mx-auto px-10 py-8">
        {/* Large title */}
        <h1 className="mb-8 text-headline-lg font-bold tracking-tight text-on-surface">
          {resolveLabel(sectionLabelKey)}
        </h1>

        {/* Settings groups as cards */}
        {groups.map((group, gi) => (
          <section key={`${group.labelKey}-${gi}`} className="mb-6">
            <h2 className="mb-2 text-body-sm font-semibold text-on-surface">
              {resolveLabel(group.labelKey)}
            </h2>
            <div className="overflow-hidden rounded-xl border border-outline bg-surface">
              {group.items.map((item, idx) => (
                <div key={item.id} className={cn(idx > 0 && "border-t border-outline")}>
                  <SettingRow
                    item={item}
                    form={form}
                    themeMode={themeMode}
                    onThemeModeChange={onThemeModeChange}
                    resolveLabel={resolveLabel}
                  />
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}

function SettingRow({
  item,
  form,
  themeMode,
  onThemeModeChange,
  resolveLabel,
}: {
  item: SettingItem;
  form: UseFormReturn<SettingsFormData>;
  themeMode: ThemeMode;
  onThemeModeChange: (mode: ThemeMode) => void;
  resolveLabel: (key: string) => string;
}) {
  const watchedValues = form.watch();
  const [langOpen, setLangOpen] = useState(false);
  const langRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!langOpen) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (langRef.current && !langRef.current.contains(e.target as Node)) {
        setLangOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [langOpen]);

  const segmentIndex = (() => {
    if (item.id === "grid-style") {
      const val = watchedValues.gridStyle;
      return val === "dots" ? 0 : val === "lines" ? 1 : 2;
    }
    if (item.id === "cmd-palette") {
      const val = watchedValues.cmdPaletteKey;
      return val === "⌘P" ? 0 : 1;
    }
    return 0;
  })();

  const isColorMode = item.id === "color-mode";
  const isGridStyle = item.id === "grid-style";
  const isCmdPalette = item.id === "cmd-palette";
  const isLanguage = item.id === "language";
  const currentSegment = isLanguage
    ? getLocale() === "zh"
      ? 1
      : 0
    : isColorMode
      ? themeMode === "dark"
        ? 1
        : 0
      : segmentIndex;

  const toggleValue = (() => {
    switch (item.id) {
      case "auto-save":
        return watchedValues.autoSave;
      case "restore-session":
        return watchedValues.restoreSession;
      case "check-updates":
        return watchedValues.checkUpdates;
      case "minimap":
        return watchedValues.minimap;
      case "auto-convert":
        return watchedValues.autoConvert;
      default:
        return false;
    }
  })();

  const selectValue = (() => {
    switch (item.id) {
      case "backend": {
        const val = watchedValues.backend;
        return val === "Burn" ? resolveLabel("settings.burn") : resolveLabel("settings.candle");
      }
      case "device": {
        const val = watchedValues.device;
        return val === "auto"
          ? resolveLabel("settings.auto-detect")
          : val === "cpu"
            ? resolveLabel("settings.cpu")
            : val === "gpu0"
              ? resolveLabel("settings.gpu-0")
              : resolveLabel("settings.gpu-1");
      }
      case "memory-budget": {
        const val = watchedValues.memoryBudget;
        return val === "2GB"
          ? resolveLabel("settings.2gb")
          : val === "4GB"
            ? resolveLabel("settings.4gb")
            : val === "8GB"
              ? resolveLabel("settings.8gb")
              : resolveLabel("settings.16gb");
      }
      case "project-dir":
        return watchedValues.projectDir;
      case "autosave-interval": {
        const val = watchedValues.autosaveInterval;
        return val === "5s"
          ? resolveLabel("settings.5-seconds")
          : val === "10s"
            ? resolveLabel("settings.10-seconds")
            : val === "30s"
              ? resolveLabel("settings.30-seconds")
              : resolveLabel("settings.1-minute");
      }
      case "download-dir":
        return watchedValues.downloadDir;
      default:
        return "";
    }
  })();

  const handleToggle = () => {
    switch (item.id) {
      case "auto-save":
        form.setValue("autoSave", !toggleValue);
        break;
      case "restore-session":
        form.setValue("restoreSession", !toggleValue);
        break;
      case "check-updates":
        form.setValue("checkUpdates", !toggleValue);
        break;
      case "minimap":
        form.setValue("minimap", !toggleValue);
        break;
      case "auto-convert":
        form.setValue("autoConvert", !toggleValue);
        break;
    }
  };

  const handleSelect = (value: string) => {
    switch (item.id) {
      case "backend":
        form.setValue("backend", value.includes("Candle") ? "Candle (deprecated)" : "Burn");
        break;
      case "device": {
        const v = value.includes("Auto")
          ? "auto"
          : value.includes("CPU")
            ? "cpu"
            : value.includes("0")
              ? "gpu0"
              : "gpu1";
        form.setValue("device", v as "auto" | "cpu" | "gpu0" | "gpu1");
        break;
      }
      case "memory-budget": {
        const v = value.includes("2")
          ? "2GB"
          : value.includes("4")
            ? "4GB"
            : value.includes("8")
              ? "8GB"
              : "16GB";
        form.setValue("memoryBudget", v as "2GB" | "4GB" | "8GB" | "16GB");
        break;
      }
      case "project-dir":
        form.setValue("projectDir", value);
        break;
      case "autosave-interval": {
        const v = value.includes("5")
          ? "5s"
          : value.includes("10")
            ? "10s"
            : value.includes("30")
              ? "30s"
              : "60s";
        form.setValue("autosaveInterval", v as "5s" | "10s" | "30s" | "60s");
        break;
      }
      case "download-dir":
        form.setValue("downloadDir", value);
        break;
    }
  };

  const handleSegment = (i: number) => {
    if (isLanguage) {
      setLocale(i === 0 ? "en" : "zh");
    } else if (isColorMode) {
      onThemeModeChange(i === 0 ? "light" : "dark");
    } else if (isGridStyle) {
      form.setValue("gridStyle", i === 0 ? "dots" : i === 1 ? "lines" : "none");
    } else if (isCmdPalette) {
      form.setValue("cmdPaletteKey", i === 0 ? "⌘P" : "⌘K");
    }
  };

  return (
    <div className="flex items-center justify-between gap-4 px-5 py-3.5">
      <div className="min-w-0 flex-1">
        <div className="text-body-sm font-medium text-on-surface">
          {resolveLabel(item.labelKey)}
        </div>
        {item.descriptionKey && (
          <div className="mt-0.5 text-caption leading-relaxed text-on-surface-variant">
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
              "relative shrink-0 rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-secondary/30",
              toggleValue ? "bg-status-ready" : "bg-control-border",
            )}
            style={{ width: "40px", height: "22px" }}
          >
            <span
              className="absolute top-[2px] rounded-full bg-white shadow-sm transition-all"
              style={{
                width: "18px",
                height: "18px",
                left: toggleValue ? "20px" : "2px",
              }}
            />
          </button>
        )}

        {item.type === "select" && (
          <select
            style={{ height: "32px", fontSize: "13px", padding: "0 10px" }}
            className="rounded-md border border-outline bg-surface text-on-surface outline-none focus-visible:border-secondary/30 focus-visible:ring-2 focus-visible:ring-secondary/10"
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

        {item.type === "segment" && item.options && isLanguage && (
          <div ref={langRef} className="relative">
            <button
              type="button"
              onClick={() => setLangOpen(!langOpen)}
              style={{ height: "32px", padding: "0 10px", fontSize: "13px" }}
              className="flex items-center gap-1.5 rounded-md border border-outline bg-surface font-medium text-on-surface hover:bg-control-hover transition-colors"
            >
              {resolveLabel(item.options[currentSegment])}
              <span className="text-on-surface-variant text-[10px]">▾</span>
            </button>
            {langOpen && (
              <div className="absolute right-0 top-full mt-1 z-50 min-w-[120px] rounded-md border border-outline bg-surface py-1 shadow-md">
                {item.options.map((opt, i) => (
                  <button
                    key={opt}
                    type="button"
                    onClick={() => {
                      handleSegment(i);
                      setLangOpen(false);
                    }}
                    className={cn(
                      "w-full text-left px-3 py-1.5 text-[13px] transition-colors",
                      currentSegment === i
                        ? "bg-surface-container-low text-on-surface font-medium"
                        : "text-on-surface hover:bg-control-hover",
                    )}
                  >
                    {resolveLabel(opt)}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {item.type === "segment" && item.options && !isLanguage && (
          <div
            className="flex rounded-md border border-outline/50 bg-surface-container-low"
            style={{ padding: "2px" }}
          >
            {item.options.map((opt, i) => (
              <button
                key={opt}
                type="button"
                onClick={() => handleSegment(i)}
                style={{ padding: "4px 12px", fontSize: "13px", lineHeight: "18px" }}
                className={cn(
                  "flex-1 rounded font-medium transition-all",
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
