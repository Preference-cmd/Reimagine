import {
  Bell,
  Check,
  Cpu,
  Folder,
  Keyboard,
  Moon,
  Palette,
  Shield,
  Sun,
  X,
} from "lucide-react";
import { Dialog, RadioGroup } from "radix-ui";
import { useState } from "react";
import { cn } from "@/lib/utils";

export type ThemeMode = "light" | "dark";

const THEME_OPTIONS: Array<{
  value: ThemeMode;
  label: string;
  description: string;
  icon: typeof Sun;
}> = [
  {
    value: "light",
    label: "Light",
    description: "Bright canvas and white panel surfaces.",
    icon: Sun,
  },
  {
    value: "dark",
    label: "Dark",
    description: "Low-light canvas with Reimagine dark primitives.",
    icon: Moon,
  },
];

const SETTINGS_SECTIONS = [
  {
    id: "appearance",
    label: "Appearance",
    description: "Theme and visual preferences.",
    icon: Palette,
    status: "ready",
  },
  {
    id: "workspace",
    label: "Workspace",
    description: "Project, files, and local workspace defaults.",
    icon: Folder,
    status: "planned",
  },
  {
    id: "runtime",
    label: "Runtime",
    description: "Backend, device, queue, and diagnostics preferences.",
    icon: Cpu,
    status: "planned",
  },
  {
    id: "shortcuts",
    label: "Shortcuts",
    description: "Keyboard bindings for graph editing and execution.",
    icon: Keyboard,
    status: "planned",
  },
  {
    id: "notifications",
    label: "Notifications",
    description: "Run completion and failure alerts.",
    icon: Bell,
    status: "planned",
  },
  {
    id: "privacy",
    label: "Privacy",
    description: "Local data, telemetry, and model metadata controls.",
    icon: Shield,
    status: "planned",
  },
] as const;

type SettingsSectionId = (typeof SETTINGS_SECTIONS)[number]["id"];

function getSection(id: SettingsSectionId) {
  return (
    SETTINGS_SECTIONS.find((section) => section.id === id) ??
    SETTINGS_SECTIONS[0]
  );
}

export function SettingsPanel({
  open,
  themeMode,
  onThemeModeChange,
  onClose,
}: {
  open: boolean;
  themeMode: ThemeMode;
  onThemeModeChange: (mode: ThemeMode) => void;
  onClose: () => void;
}) {
  const [activeSection, setActiveSection] =
    useState<SettingsSectionId>("appearance");

  const section = getSection(activeSection);

  return (
    <Dialog.Root open={open} onOpenChange={(nextOpen) => !nextOpen && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-[var(--overlay-z-modal-backdrop)] bg-background/18 backdrop-blur-[2px]" />
        <Dialog.Content
          aria-label="Settings"
          onCloseAutoFocus={(event) => {
            const settingsTrigger = document.querySelector<HTMLElement>(
              '[data-shell-panel="Settings"]',
            );
            if (!settingsTrigger) return;

            event.preventDefault();
            settingsTrigger.focus();
          }}
          className="fixed left-1/2 top-1/2 z-[var(--overlay-z-modal)] flex h-[min(560px,calc(100vh-48px))] w-[min(720px,calc(100vw-48px))] -translate-x-1/2 -translate-y-1/2 overflow-hidden rounded-xl border border-outline bg-surface shadow-modal outline-none"
        >
          <aside className="flex w-48 shrink-0 flex-col border-r border-outline bg-surface-container-lowest">
            <div className="flex h-12 items-center justify-between border-b border-outline px-3">
              <Dialog.Title className="text-body-sm font-semibold text-on-surface">
                Settings
              </Dialog.Title>
            </div>

            <nav
              className="flex-1 space-y-0.5 p-2"
              aria-label="Settings sections"
            >
              {SETTINGS_SECTIONS.map((section) => {
                const Icon = section.icon;
                const active = section.id === activeSection;

                return (
                  <button
                    key={section.id}
                    aria-current={active ? "page" : undefined}
                    onClick={() => setActiveSection(section.id)}
                    className={cn(
                      "flex min-h-8 w-full cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-left text-body-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30",
                      active
                        ? "bg-surface-container-high text-on-surface"
                        : "text-on-surface-variant hover:bg-control-hover hover:text-on-surface",
                    )}
                    type="button"
                  >
                    <Icon className="h-3.5 w-3.5 shrink-0" />
                    <span className="truncate">{section.label}</span>
                  </button>
                );
              })}
            </nav>
          </aside>

          <main className="flex min-w-0 flex-1 flex-col bg-surface">
            <header className="flex h-12 items-center justify-between border-b border-outline px-4">
              <div className="min-w-0">
                <h2 className="truncate text-body-sm font-semibold text-on-surface">
                  {section.label}
                </h2>
                <Dialog.Description className="text-caption text-on-surface-variant">
                  {section.description}
                </Dialog.Description>
              </div>
              <Dialog.Close asChild>
                <button
                  aria-label="Close settings"
                  className="rounded-md p-1 text-on-surface-variant hover:bg-control-hover hover:text-on-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30"
                  type="button"
                >
                  <X className="h-4 w-4" />
                </button>
              </Dialog.Close>
            </header>

            <div className="scrollbar-hide min-w-0 flex-1 overflow-y-auto p-5">
              {activeSection === "appearance" ? (
                <AppearanceSettings
                  themeMode={themeMode}
                  onThemeModeChange={onThemeModeChange}
                />
              ) : (
                <PlannedSettings section={section} />
              )}
            </div>
          </main>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function AppearanceSettings({
  themeMode,
  onThemeModeChange,
}: {
  themeMode: ThemeMode;
  onThemeModeChange: (mode: ThemeMode) => void;
}) {
  return (
    <div className="w-full min-w-0 space-y-4">
      <SettingsSection
        title="Color mode"
        description="Choose how the editor surface renders during long graph sessions."
      >
        <RadioGroup.Root
          className="grid w-full min-w-0 grid-cols-[repeat(2,minmax(0,1fr))] gap-2"
          aria-label="Theme mode"
          value={themeMode}
          onValueChange={(value) => onThemeModeChange(value as ThemeMode)}
        >
          {THEME_OPTIONS.map((option) => {
            const Icon = option.icon;
            const selected = option.value === themeMode;

            return (
              <RadioGroup.Item
                key={option.value}
                className={cn(
                  "flex min-h-16 w-full min-w-0 cursor-pointer items-start gap-2.5 rounded-lg border p-2.5 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30",
                  selected
                    ? "border-primary/30 bg-control-hover text-on-surface"
                    : "border-outline bg-surface text-on-surface-variant hover:bg-control-hover hover:text-on-surface",
                )}
                value={option.value}
              >
                <span
                  className={cn(
                    "flex h-7 w-7 shrink-0 items-center justify-center rounded-md border",
                    selected
                      ? "border-primary/20 bg-surface text-primary"
                      : "border-outline bg-surface-container-low text-on-surface-variant",
                  )}
                >
                  <Icon className="h-3.5 w-3.5" />
                </span>
                <span className="min-w-0 flex-1">
                  <span className="flex min-w-0 items-center gap-2 text-body-sm font-semibold">
                    {option.label}
                    {selected && <Check className="h-3.5 w-3.5 text-primary" />}
                  </span>
                  <span className="mt-1 block text-caption text-on-surface-variant">
                    {option.description}
                  </span>
                </span>
              </RadioGroup.Item>
            );
          })}
        </RadioGroup.Root>
      </SettingsSection>

      <SettingsSection title="Behavior">
        <SettingsFact
          label="Persistence"
          value="Theme is stored locally on this device."
        />
        <SettingsFact
          label="Canvas"
          value="Canvas follows the selected mode."
        />
      </SettingsSection>
    </div>
  );
}

function PlannedSettings({
  section,
}: {
  section: (typeof SETTINGS_SECTIONS)[number];
}) {
  const Icon = section.icon;
  const plannedItems = getPlannedItems(section.id);

  return (
    <div className="w-full min-w-0 space-y-4">
      <section className="rounded-lg border border-dashed border-outline bg-surface-container-low/55 p-3.5">
        <div className="flex items-start gap-2.5">
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-outline bg-surface text-on-surface-variant">
            <Icon className="h-3.5 w-3.5" />
          </span>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h3 className="text-body-sm font-semibold text-on-surface">
                {section.label}
              </h3>
              <span className="rounded-full bg-surface-container-high px-2 py-0.5 text-caption font-medium text-on-surface-variant">
                Planned
              </span>
            </div>
            <p className="mt-0.5 text-caption text-on-surface-variant">
              {section.description}
            </p>
          </div>
        </div>
      </section>

      <SettingsSection title="Expected controls">
        <div className="divide-y divide-outline">
          {plannedItems.map((item) => (
            <div
              key={item}
              className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-4 py-2.5"
            >
              <span className="min-w-0 text-body-sm text-on-surface">
                {item}
              </span>
              <span className="shrink-0 text-caption text-on-surface-variant">
                Planned
              </span>
            </div>
          ))}
        </div>
      </SettingsSection>
    </div>
  );
}

function SettingsSection({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="overflow-hidden rounded-lg border border-outline bg-surface">
      <div className="border-b border-outline px-3.5 py-2.5">
        <h3 className="text-body-sm font-semibold text-on-surface">{title}</h3>
        {description && (
          <p className="mt-0.5 max-w-[560px] text-caption text-on-surface-variant">
            {description}
          </p>
        )}
      </div>
      <div className="p-3.5">{children}</div>
    </section>
  );
}

function SettingsFact({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid grid-cols-[120px_minmax(0,1fr)] gap-4 border-b border-outline py-2.5 text-body-sm last:border-b-0 first:pt-0 last:pb-0">
      <span className="text-on-surface-variant">{label}</span>
      <span className="min-w-0 text-on-surface">{value}</span>
    </div>
  );
}

function getPlannedItems(section: SettingsSectionId): string[] {
  switch (section) {
    case "workspace":
      return [
        "Default project folder",
        "Model and asset index locations",
        "Autosave and recovery behavior",
      ];
    case "runtime":
      return [
        "Preferred inference backend",
        "Device selection and memory budget",
        "Run queue and diagnostics verbosity",
      ];
    case "shortcuts":
      return [
        "Command palette shortcut",
        "Graph navigation and node editing bindings",
        "Run, stop, save, and export bindings",
      ];
    case "notifications":
      return [
        "Run completion alerts",
        "Failure and missing model warnings",
        "Long-running workflow notifications",
      ];
    case "privacy":
      return [
        "Local-only metadata policy",
        "Crash and diagnostics sharing",
        "Model path redaction in exported logs",
      ];
    default:
      return [];
  }
}
