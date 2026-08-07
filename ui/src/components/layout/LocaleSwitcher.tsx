import { setLocale, getLocale } from "$paraglide/runtime";
import type { Locale } from "$paraglide/runtime";

const LOCALES: { code: Locale; label: string }[] = [
  { code: "en", label: "English" },
  { code: "zh", label: "中文" },
];

export function LocaleSwitcher() {
  const current = getLocale();

  return (
    <div className="flex gap-1 rounded-lg bg-surface-container-low p-0.5">
      {LOCALES.map(({ code, label }) => (
        <button
          key={code}
          type="button"
          onClick={() => setLocale(code)}
          className={`rounded-md px-3 py-1 text-caption font-medium transition-colors ${
            current === code
              ? "bg-surface text-on-surface shadow-sm"
              : "text-on-surface-variant hover:text-on-surface"
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
