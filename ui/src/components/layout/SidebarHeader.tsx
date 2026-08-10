import { ChevronDown, Search } from "lucide-react";
import { useUIStore } from "@/store/uiStore";

function Logo({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="Reimagine logo"
    >
      <path
        d="M16 4C10.5 4 6.5 8.5 6.5 13.5C6.5 18 9 21.5 12.5 24.5C14 25.8 15 27 16 29C17 27 18 25.8 19.5 24.5C23 21.5 25.5 18 25.5 13.5C25.5 8.5 21.5 4 16 4Z"
        fill="currentColor"
      />
      <ellipse cx="16" cy="13.5" rx="4" ry="5.5" fill="#111111" />
    </svg>
  );
}

export function SidebarHeader({ collapsed }: { collapsed?: boolean }) {
  const openCommandPalette = useUIStore((s) => s.openCommandPalette);

  return (
    <div className="flex items-center gap-2 px-3 py-3">
      <Logo className="h-5 w-5 shrink-0 text-sidebar-text-secondary" />

      {!collapsed && (
        <button
          type="button"
          className="flex items-center gap-1 text-sm font-medium text-sidebar-text-primary hover:text-white focus-visible:outline-none"
        >
          Reimagine
          <ChevronDown className="h-3 w-3 text-sidebar-text-muted" />
        </button>
      )}

      <button
        type="button"
        aria-label="Search"
        onClick={openCommandPalette}
        className="ml-auto flex h-7 w-7 shrink-0 cursor-pointer items-center justify-center rounded-md text-sidebar-text-muted transition-colors hover:bg-sidebar-item-hover hover:text-sidebar-text-secondary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/20"
      >
        <Search className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
