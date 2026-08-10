import { createFileRoute } from "@tanstack/react-router";
import { SettingsView } from "@/components/layout/SettingsView";
import { useTheme } from "@/lib/theme";

function SettingsRoute() {
  const { themeMode, onThemeModeChange } = useTheme();
  return <SettingsView themeMode={themeMode} onThemeModeChange={onThemeModeChange} />;
}

export const Route = createFileRoute("/settings")({
  component: SettingsRoute,
});
