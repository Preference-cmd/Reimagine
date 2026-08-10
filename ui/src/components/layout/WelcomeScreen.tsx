import {
  ImageIcon,
  Sparkles,
  Paintbrush,
  ArrowUpRight,
  Plus,
  Paperclip,
  ChevronDown,
  Send,
} from "lucide-react";
import { cn } from "@/lib/utils";
import * as m from "$paraglide/messages";

/**
 * Welcome screen — shown when starting a new project.
 * Codex-style layout: title + input bar + action cards.
 */

type ActionCard = {
  icon: React.ComponentType<{ className?: string }>;
  labelKey: string;
  descKey: string;
};

const ACTION_CARDS: ActionCard[] = [
  { icon: ImageIcon, labelKey: "welcome.textToImage", descKey: "welcome.textToImageDesc" },
  { icon: Sparkles, labelKey: "welcome.imageToImage", descKey: "welcome.imageToImageDesc" },
  { icon: Paintbrush, labelKey: "welcome.inpaint", descKey: "welcome.inpaintDesc" },
  { icon: ArrowUpRight, labelKey: "welcome.upscale", descKey: "welcome.upscaleDesc" },
];

function Card({ icon: Icon, labelKey, descKey }: ActionCard) {
  const label = (m as unknown as Record<string, () => string>)[labelKey]();
  const desc = (m as unknown as Record<string, () => string>)[descKey]();

  return (
    <button
      type="button"
      className={cn(
        "group flex items-start gap-3 rounded-lg border border-outline/50 p-4 text-left transition-all duration-150",
        "bg-surface hover:border-outline hover:bg-surface-container-high",
        "hover:shadow-[0_2px_8px_-2px_rgb(0_0_0/0.08)]",
        "cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30",
      )}
    >
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-surface-container-high text-on-surface-variant transition-colors group-hover:bg-surface-container">
        <Icon className="h-4 w-4" />
      </div>
      <div className="min-w-0">
        <div className="text-body-sm font-medium text-on-surface">{label}</div>
        <div className="mt-0.5 text-caption text-on-surface-variant">{desc}</div>
      </div>
    </button>
  );
}

export function WelcomeScreen() {
  return (
    <div className="flex h-full flex-col items-center justify-center bg-background px-8">
      <div className="flex w-full max-w-[680px] flex-col items-center">
        {/* Title */}
        <h1 className="mb-2 text-display-sm font-semibold tracking-tight text-on-surface">
          {m["welcome.title"]()}
        </h1>
        <p className="mb-8 text-body-md text-on-surface-variant">{m["welcome.subtitle"]()}</p>

        {/* Agent input box */}
        <div className="mb-10 w-full">
          <div className="rounded-xl border border-outline bg-surface shadow-sm transition-colors focus-within:border-primary/40 focus-within:ring-2 focus-within:ring-primary/10">
            {/* Textarea */}
            <textarea
              placeholder={m["welcome.inputPlaceholder"]()}
              rows={3}
              className="w-full resize-none bg-transparent px-4 pt-3 pb-2 text-body-sm text-on-surface outline-none placeholder:text-on-surface-variant/50"
            />

            {/* Bottom toolbar */}
            <div className="flex items-center justify-between border-t border-outline/50 px-3 py-2">
              <div className="flex items-center gap-1">
                {/* Attach button */}
                <button
                  type="button"
                  className="flex h-7 w-7 items-center justify-center rounded-md text-on-surface-variant transition-colors hover:bg-control-hover"
                  aria-label="Attach file"
                >
                  <Plus className="h-4 w-4" />
                </button>
                {/* Add resource */}
                <button
                  type="button"
                  className="flex items-center gap-1.5 rounded-md px-2 py-1 text-caption text-on-surface-variant transition-colors hover:bg-control-hover"
                >
                  <Paperclip className="h-3.5 w-3.5" />
                  <span>Add resource</span>
                </button>
              </div>

              <div className="flex items-center gap-1.5">
                {/* Model selector */}
                <button
                  type="button"
                  className="flex items-center gap-1 rounded-md px-2 py-1 text-caption text-on-surface-variant transition-colors hover:bg-control-hover"
                >
                  <span>Burn</span>
                  <ChevronDown className="h-3 w-3" />
                </button>
                {/* Send button */}
                <button
                  type="button"
                  className="flex h-7 w-7 items-center justify-center rounded-full bg-on-surface text-background transition-opacity hover:opacity-90"
                  aria-label="Send"
                >
                  <Send className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Get Started section */}
        <div className="w-full">
          <h2 className="mb-3 text-label-caps text-on-surface-variant/60">
            {m["welcome.getStarted"]()}
          </h2>
          <div className="grid grid-cols-2 gap-3">
            {ACTION_CARDS.map((card) => (
              <Card key={card.labelKey} {...card} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
