import {
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  Download,
  Play,
  Save,
  Search,
  Share2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { RuntimeIsland } from "./RuntimeIsland";
import { useRuntimeStore } from "@/store/runtime";

function Logo({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={cn("h-6 w-6", className)}
      aria-label="Reimagine logo"
    >
      <path
        d="M16 4C10.5 4 6.5 8.5 6.5 13.5C6.5 18 9 21.5 12.5 24.5C14 25.8 15 27 16 29C17 27 18 25.8 19.5 24.5C23 21.5 25.5 18 25.5 13.5C25.5 8.5 21.5 4 16 4Z"
        fill="currentColor"
      />
      <ellipse cx="16" cy="13.5" rx="4" ry="5.5" fill="#131313" />
    </svg>
  );
}

function TopBarButton({
  children,
  ariaLabel,
  variant = "ghost",
  disabled,
  onClick,
  className,
}: {
  children: React.ReactNode;
  ariaLabel: string;
  variant?: "ghost" | "primary";
  disabled?: boolean;
  onClick?: () => void;
  className?: string;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      aria-label={ariaLabel}
      className={cn(
        "flex h-8 w-8 cursor-pointer items-center justify-center rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30 focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:cursor-default disabled:opacity-45",
        variant === "primary"
          ? "bg-primary text-on-primary hover:bg-primary/90"
          : "text-on-surface-variant hover:bg-control-hover hover:text-on-surface",
        className,
      )}
    >
      {children}
    </button>
  );
}

function ProjectSelector({ name }: { name: string }) {
  return (
    <button
      type="button"
      aria-label={`Switch project, current project ${name}`}
      className="panel-flat ml-1.5 flex h-11 min-w-0 cursor-pointer items-center gap-1.5 rounded-2xl px-sm text-body-sm font-medium text-on-surface-variant transition-colors hover:bg-control-hover hover:text-on-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30 focus-visible:ring-offset-2 focus-visible:ring-offset-background"
    >
      <span className="truncate max-w-36">{name}</span>
      <ChevronDown className="h-3.5 w-3.5 shrink-0" />
    </button>
  );
}

export function TopBar({
  forceRuntimeCollapsed = false,
}: {
  forceRuntimeCollapsed?: boolean;
}) {
  const startMockRun = useRuntimeStore((s) => s.startMockRun);
  const phase = useRuntimeStore((s) => s.phase);
  const runActive = phase === "starting" || phase === "running";

  return (
    <div className="pointer-events-auto absolute inset-x-0 top-0 z-[var(--overlay-z-topbar)] flex flex-col gap-2 px-md py-2.5">
      <header className="relative flex h-11 items-center">
        <div className="panel-flat z-20 flex h-11 w-11 items-center justify-center rounded-2xl text-on-surface">
          <Logo className="h-5 w-5" />
        </div>

        <ProjectSelector name="Black bear" />

        <TopBarButton
          ariaLabel="Search commands"
          className="panel-flat ml-1.5 h-11 w-11 rounded-2xl"
        >
          <Search className="h-4 w-4" />
        </TopBarButton>

        <div className="pointer-events-auto absolute left-1/2 top-0 z-30 -translate-x-1/2">
          <RuntimeIsland forceCollapsed={forceRuntimeCollapsed} />
        </div>

        <div className="panel-flat z-20 ml-auto flex h-11 items-center gap-1 rounded-2xl pr-1.5">
          <TopBarButton ariaLabel="Go back">
            <ChevronLeft className="h-4 w-4" />
          </TopBarButton>
          <TopBarButton ariaLabel="Go forward">
            <ChevronRight className="h-4 w-4" />
          </TopBarButton>
          <TopBarButton ariaLabel="Save workflow">
            <Save className="h-4 w-4" />
          </TopBarButton>
          <TopBarButton ariaLabel="Export workflow">
            <Download className="h-4 w-4" />
          </TopBarButton>
          <TopBarButton ariaLabel="Share workflow">
            <Share2 className="h-4 w-4" />
          </TopBarButton>
          <TopBarButton
            ariaLabel="Run workflow"
            disabled={runActive}
            onClick={startMockRun}
            variant="primary"
          >
            <Play className="h-4 w-4 fill-current" />
          </TopBarButton>
        </div>
      </header>

      <div className="flex items-start justify-end px-xs">
        <span className="text-caption font-medium text-on-surface-variant/70">
          image generation v.3
        </span>
      </div>
    </div>
  );
}
