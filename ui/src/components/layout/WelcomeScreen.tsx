import { Compass, Puzzle, Eye, Bug } from "lucide-react";
import { cn } from "@/lib/utils";
import { useWorkflowStore } from "@/store/workflow";

/**
 * Codex-style welcome screen — shown when no chat is active.
 * Centered hero with 4 action cards + input bar at bottom.
 */

type ActionCard = {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  color: string;
};

const ACTION_CARDS: ActionCard[] = [
  { icon: Compass, label: "探索并理解代码", color: "text-emerald-500" },
  { icon: Puzzle, label: "构建新功能、应用模式工具", color: "text-violet-500" },
  { icon: Eye, label: "审查代码并提出修改建议", color: "text-amber-500" },
  { icon: Bug, label: "修复问题和失败", color: "text-rose-500" },
];

/** Abstract cloud/logo icon — placeholder for the Reimagine brand mark. */
function BrandIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
    >
      <path
        d="M20 44c-6.627 0-12-5.373-12-12 0-5.89 4.244-10.813 9.857-11.76C19.366 13.18 24.34 8 30.5 8c6.16 0 11.134 5.18 12.643 12.24C48.756 21.187 53 26.11 53 32c0 6.627-5.373 12-12 12H20z"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinejoin="round"
        fill="none"
      />
      <path
        d="M26 40l4-6 4 6"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="38" cy="34" r="1.5" fill="currentColor" />
    </svg>
  );
}

function Card({ icon: Icon, label, color }: ActionCard) {
  return (
    <button
      type="button"
      className={cn(
        "group flex flex-col items-center gap-3 rounded-xl border border-outline/50",
        "bg-surface px-5 py-5 text-center transition-all duration-150",
        "hover:border-outline hover:bg-surface-container-high",
        "hover:shadow-[0_2px_8px_-2px_rgb(0_0_0/0.08)]",
        "cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30",
        "w-[160px]",
      )}
    >
      <Icon className={cn("h-6 w-6", color)} />
      <span className="text-body-sm leading-snug text-on-surface">{label}</span>
    </button>
  );
}

export function WelcomeScreen() {
  const workflowName = useWorkflowStore((s) => s.name);

  return (
    <div className="flex h-full flex-col items-center justify-center bg-background">
      {/* Hero */}
      <div className="flex flex-col items-center gap-4 pb-8">
        <BrandIcon className="h-14 w-14 text-on-surface/30" />
        <h1 className="text-display-md font-semibold tracking-tight text-on-surface">
          我们应该在 Reimagine 中构建什么？
        </h1>
      </div>

      {/* Action cards */}
      <div className="flex gap-3 pb-10">
        {ACTION_CARDS.map((card) => (
          <Card key={card.label} {...card} />
        ))}
      </div>

      {/* Input bar */}
      <div className="w-full max-w-[640px] px-4">
        <div className="flex flex-col rounded-xl border border-outline bg-surface shadow-sm">
          {/* Context chips */}
          <div className="flex items-center gap-2 border-b border-outline px-3 py-2">
            <span className="flex items-center gap-1 rounded-md bg-surface-container-high px-2 py-0.5 text-caption font-medium text-on-surface-variant">
              <svg
                viewBox="0 0 16 16"
                className="h-3.5 w-3.5"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
              >
                <rect x="2" y="3" width="12" height="10" rx="2" />
                <path d="M5 7h6M5 10h3" />
              </svg>
              {workflowName}
            </span>
            <span className="flex items-center gap-1 rounded-md bg-surface-container-high px-2 py-0.5 text-caption font-medium text-on-surface-variant">
              <svg
                viewBox="0 0 16 16"
                className="h-3.5 w-3.5"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
              >
                <path d="M2 4h12M2 8h12M2 12h8" />
              </svg>
              本地
            </span>
            <span className="flex items-center gap-1 rounded-md bg-surface-container-high px-2 py-0.5 text-caption font-medium text-on-surface-variant">
              <svg
                viewBox="0 0 16 16"
                className="h-3.5 w-3.5"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
              >
                <circle cx="8" cy="8" r="3" />
              </svg>
              main
            </span>
          </div>

          {/* Textarea */}
          <textarea
            placeholder="场合输入"
            rows={2}
            className="resize-none bg-transparent px-3 py-2.5 text-body-sm text-on-surface outline-none placeholder:text-on-surface-variant/50"
          />

          {/* Bottom bar */}
          <div className="flex items-center justify-between border-t border-outline px-3 py-2">
            <div className="flex items-center gap-2">
              <button
                type="button"
                className="flex h-7 w-7 items-center justify-center rounded-md text-on-surface-variant transition-colors hover:bg-control-hover"
                aria-label="Add attachment"
              >
                <svg
                  viewBox="0 0 16 16"
                  className="h-4 w-4"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                >
                  <path d="M8 3v10M3 8h10" />
                </svg>
              </button>
              <span className="text-caption text-status-success">完全访问</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-caption tabular-nums text-on-surface-variant">5.6 Sel</span>
              <span className="text-caption text-on-surface-variant">中</span>
              <button
                type="button"
                className="flex h-7 w-7 items-center justify-center rounded-md text-on-surface-variant transition-colors hover:bg-control-hover"
                aria-label="Voice input"
              >
                <svg
                  viewBox="0 0 16 16"
                  className="h-4 w-4"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                >
                  <rect x="6" y="2" width="4" height="7" rx="2" />
                  <path d="M4 7a4 4 0 008 0M8 11v2" />
                </svg>
              </button>
              <button
                type="button"
                className="flex h-7 w-7 items-center justify-center rounded-full bg-on-surface text-background transition-opacity hover:opacity-90"
                aria-label="Send"
              >
                <svg
                  viewBox="0 0 16 16"
                  className="h-3.5 w-3.5"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <path d="M3 8h10M9 4l4 4-4 4" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
