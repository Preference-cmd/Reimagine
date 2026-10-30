import type { ComponentProps } from "react";
import { useEffect, useState } from "react";
import { useRouter, useRouterState } from "@tanstack/react-router";
import { cn } from "@/lib/utils";
import { isTauri } from "@/ipc/commands";

/** True inside the Tauri webview; false in the plain-browser review surface. */
export function isTauriApp(): boolean {
  return isTauri();
}

/** macOS keeps the native traffic lights in Tauri Overlay mode. */
export function isMacPlatform(): boolean {
  if (typeof navigator === "undefined") return false;
  return /mac/i.test(navigator.userAgent);
}

/** The floating session panel belongs to the canvas and library modes only. */
export function showsSessionPanel(pathname: string): boolean {
  return pathname === "/" || pathname === "/library";
}

/** Shared 32px navigation row used by the sidebar. */
export const SHELL_NAV_ITEM =
  "flex h-8 w-full cursor-pointer items-center gap-2.5 rounded-sm px-2.5 text-left text-ui text-fg transition-colors duration-150 hover:bg-hover active:bg-press focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:cursor-default disabled:opacity-40 disabled:hover:bg-transparent";

/**
 * Decorative traffic lights for the browser preview. The desktop app draws
 * the real ones, so this must stay hidden inside Tauri.
 */
export function WindowLights({ className }: { className?: string }) {
  return (
    <span aria-hidden="true" className={cn("flex shrink-0 items-center gap-2 px-1", className)}>
      <span className="block size-3 rounded-full bg-window-close shadow-[inset_0_0_0_0.5px_var(--color-window-light-ring)]" />
      <span className="block size-3 rounded-full bg-window-minimize shadow-[inset_0_0_0_0.5px_var(--color-window-light-ring)]" />
      <span className="block size-3 rounded-full bg-window-maximize shadow-[inset_0_0_0_0.5px_var(--color-window-light-ring)]" />
    </span>
  );
}

type ShellIconButtonProps = ComponentProps<"button"> & { label: string };

/** 28px square icon button; every icon control in the shell needs a label. */
export function ShellIconButton({ label, className, children, ...props }: ShellIconButtonProps) {
  return (
    <button
      type="button"
      aria-label={label}
      className={cn(
        "flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-sm text-fg2 transition-[background-color,color,transform] duration-150 ease-ui hover:bg-hover hover:text-fg active:scale-[0.94] active:bg-press focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:cursor-default disabled:opacity-35 disabled:hover:bg-transparent disabled:hover:text-fg2 disabled:active:scale-100",
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}

export type ShellHistory = {
  canGoBack: boolean;
  canGoForward: boolean;
  goBack: () => void;
  goForward: () => void;
};

/**
 * Back/forward availability over the router history. TanStack keeps the
 * position in `location.state.__TSR_index`; a local high-water mark tracks
 * whether anything is still ahead of us. A push discards the old forward
 * entries, so the mark drops back to the new position.
 */
export function useShellHistory(): ShellHistory {
  const router = useRouter();
  const index = useRouterState({ select: (state) => state.location.state.__TSR_index });
  const [maxIndex, setMaxIndex] = useState(index);

  useEffect(() => {
    return router.history.subscribe(({ action }) => {
      const current = router.history.location.state.__TSR_index;
      setMaxIndex((previous) => (action.type === "PUSH" ? current : Math.max(previous, current)));
    });
  }, [router]);

  return {
    canGoBack: index > 0,
    canGoForward: index < maxIndex,
    goBack: () => router.history.back(),
    goForward: () => router.history.forward(),
  };
}

const RELATIVE_UNITS: Array<[Intl.RelativeTimeFormatUnit, number]> = [
  ["year", 365 * 24 * 60 * 60],
  ["month", 30 * 24 * 60 * 60],
  ["week", 7 * 24 * 60 * 60],
  ["day", 24 * 60 * 60],
  ["hour", 60 * 60],
  ["minute", 60],
];

/** Localized "3 days ago"-style label; empty for an unparseable timestamp. */
export function formatRelativeTime(iso: string, locale: string): string {
  const timestamp = Date.parse(iso);
  if (Number.isNaN(timestamp)) return "";

  const deltaSeconds = Math.round((timestamp - Date.now()) / 1000);
  const magnitude = Math.abs(deltaSeconds);
  const formatter = new Intl.RelativeTimeFormat(locale, { numeric: "auto", style: "narrow" });

  for (const [unit, seconds] of RELATIVE_UNITS) {
    if (magnitude >= seconds) {
      return formatter.format(Math.round(deltaSeconds / seconds), unit);
    }
  }
  return formatter.format(0, "second");
}
