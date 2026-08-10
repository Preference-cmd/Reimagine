import { createContext, useContext, useEffect, useState } from "react";

export type ThemeMode = "light" | "dark";

const ThemeContext = createContext<{
  themeMode: ThemeMode;
  onThemeModeChange: (mode: ThemeMode) => void;
}>({ themeMode: "light", onThemeModeChange: () => {} });

const THEME_STORAGE_KEY = "reimagine.theme";

function readStoredTheme(): ThemeMode {
  if (typeof window === "undefined") return "light";
  const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
  return stored === "dark" ? "dark" : "light";
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [themeMode, setThemeMode] = useState<ThemeMode>(readStoredTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = themeMode;
    document.documentElement.classList.toggle("dark", themeMode === "dark");
    window.localStorage.setItem(THEME_STORAGE_KEY, themeMode);
  }, [themeMode]);

  const handleThemeModeChange = (mode: ThemeMode) => {
    setThemeMode(mode);
  };

  return (
    <ThemeContext.Provider value={{ themeMode, onThemeModeChange: handleThemeModeChange }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}
