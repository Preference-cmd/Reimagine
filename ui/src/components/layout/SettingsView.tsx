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
  type: "toggle" | "select" | "segment" | "readonly";
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
          type: "readonly",
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
  modelManagement: [
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

type SettingsNavItem = { id: string; labelKey: string };
type SettingsNavGroup = { labelKey: string; items: SettingsNavItem[] };

const SETTINGS_NAV: SettingsNavGroup[] = [
  {
    labelKey: "settings.groupPersonal",
    items: [
      { id: "general", labelKey: "settings.general" },
      { id: "appearance", labelKey: "settings.appearance" },
      { id: "shortcuts", labelKey: "settings.shortcuts" },
      { id: "runtime", labelKey: "settings.runtime" },
      { id: "workspace", labelKey: "settings.workspace" },
      { id: "modelManagement", labelKey: "settings.modelManagement" },
    ],
  },
  {
    labelKey: "settings.groupIntegration",
    items: [
      { id: "plugins", labelKey: "settings.plugins" },
      { id: "browser", labelKey: "settings.browser" },
    ],
  },
  {
    labelKey: "settings.groupCoding",
    items: [{ id: "hooks", labelKey: "settings.hooks" }],
  },
];

const SECTION_LABEL_KEYS: Record<string, string> = {
  general: "settings.general",
  appearance: "settings.appearance",
  shortcuts: "settings.shortcuts",
  runtime: "settings.runtime",
  workspace: "settings.workspace",
  modelManagement: "settings.modelManagement",
  plugins: "settings.plugins",
  browser: "settings.browser",
  hooks: "settings.hooks",
};

type SettingsViewProps = {
  themeMode: ThemeMode;
  onThemeModeChange: (mode: ThemeMode) => void;
};

/**
 * SettingsView — a grouped section rail beside card sections. The active
 * section lives in the UI store so it survives route changes.
 */
export function SettingsView({ themeMode, onThemeModeChange }: SettingsViewProps) {
  const settingsNavId = useUIStore((s) => s.settingsNavId);
  const setSettingsNavId = useUIStore((s) => s.setSettingsNavId);
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
    <div className="flex min-h-0 flex-1 overflow-hidden bg-ground">
      <nav
        aria-label={resolveLabel("shell.settings")}
        className="w-[200px] shrink-0 overflow-y-auto border-r border-hair-soft px-3 py-6"
      >
        {SETTINGS_NAV.map((group) => (
          <div key={group.labelKey} className="mb-5 last:mb-0">
            <div className="px-2.5 pb-1.5 text-xs font-medium tracking-[0.02em] text-fg4">
              {resolveLabel(group.labelKey)}
            </div>
            <div className="space-y-0.5">
              {group.items.map((item) => {
                const active = item.id === activeSection;
                return (
                  <button
                    key={item.id}
                    type="button"
                    aria-current={active ? "page" : undefined}
                    onClick={() => setSettingsNavId(item.id)}
                    className={cn(
                      "flex h-8 w-full cursor-pointer items-center rounded-sm px-2.5 text-left text-ui transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent",
                      active
                        ? "bg-active font-medium text-fg"
                        : "text-fg2 hover:bg-hover hover:text-fg",
                    )}
                  >
                    <span className="min-w-0 flex-1 truncate">{resolveLabel(item.labelKey)}</span>
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      <div className="min-h-0 min-w-0 flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-[760px] px-7 py-8">
          {/* Large title */}
          <h1 className="mb-8 text-headline-lg text-fg">{resolveLabel(sectionLabelKey)}</h1>

          {/* Settings groups as cards */}
          {groups.length === 0 ? (
            <div className="rounded-xl border border-hair bg-surface px-5 py-10 text-center text-ui text-fg3">
              {resolveLabel("common.empty")}
            </div>
          ) : (
            groups.map((group, gi) => (
              <section key={`${group.labelKey}-${gi}`} className="mb-6">
                <h2 className="mb-2 text-body-sm font-semibold text-fg">
                  {resolveLabel(group.labelKey)}
                </h2>
                <div className="overflow-hidden rounded-xl border border-hair bg-surface">
                  {group.items.map((item, idx) => (
                    <div key={item.id} className={cn(idx > 0 && "border-t border-hair-soft")}>
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
            ))
          )}
        </div>
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
        <div className="text-body-sm font-medium text-fg">{resolveLabel(item.labelKey)}</div>
        {item.descriptionKey && (
          <div className="mt-0.5 text-caption leading-relaxed text-fg3">
            {resolveLabel(item.descriptionKey)}
          </div>
        )}
      </div>

      <div className="shrink-0">
        {item.type === "readonly" && (
          <div
            style={{ height: "32px", fontSize: "13px", padding: "0 10px" }}
            className="inline-flex items-center rounded-md border border-hair bg-app text-fg2 font-medium select-none"
          >
            {resolveLabel("settings.burn")}
          </div>
        )}

        {item.type === "toggle" && (
          <button
            type="button"
            role="switch"
            aria-checked={toggleValue ? "true" : "false"}
            onClick={handleToggle}
            className={cn(
              "relative shrink-0 rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent",
              toggleValue ? "bg-accent" : "bg-hair-strong",
            )}
            style={{ width: "40px", height: "22px" }}
          >
            <span
              className="absolute top-[2px] rounded-full bg-surface shadow-[0_0_0_1px_var(--color-hair-strong),var(--shadow-1)] transition-all"
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
            className="rounded-md border border-hair bg-surface text-fg outline-none focus-visible:border-accent-line focus-visible:ring-2 focus-visible:ring-accent/20"
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
              className="flex items-center gap-1.5 rounded-md border border-hair bg-surface font-medium text-fg transition-colors hover:bg-hover"
            >
              {resolveLabel(item.options[currentSegment])}
              <span className="text-[10px] text-fg3">▾</span>
            </button>
            {langOpen && (
              <div className="absolute right-0 top-full mt-1 z-50 min-w-[120px] rounded-md border border-hair bg-surface py-1 shadow-[var(--shadow-3)]">
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
                        ? "bg-active text-fg font-medium"
                        : "text-fg hover:bg-hover",
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
            className="flex rounded-md border border-hair-soft bg-app"
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
                    ? "bg-surface text-fg shadow-[var(--shadow-1)]"
                    : "text-fg3 hover:bg-hover hover:text-fg",
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
