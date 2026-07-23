import {
  Workflow,
  Boxes,
  History,
  Image,
  Settings,
} from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

type RailItem = {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  description?: string;
};

const PRIMARY_ITEMS: RailItem[] = [
  { icon: Workflow, label: "Graph", description: "Open graph explorer" },
  { icon: Boxes, label: "Models", description: "Open model library" },
  { icon: History, label: "Runs", description: "Open run history" },
  { icon: Image, label: "Assets", description: "Open asset browser" },
];

export type SideRailProps = {
  activePanel: string | null;
  onPanelChange: (panel: string | null) => void;
};

function RailButton({
  icon: Icon,
  label,
  description,
  active,
  onClick,
}: RailItem & { active: boolean; onClick: () => void }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          aria-label={label}
          aria-current={active ? "page" : undefined}
          aria-pressed={active}
          data-shell-panel={label}
          onClick={onClick}
          className={cn(
            "group relative flex h-9 w-9 cursor-pointer items-center justify-center rounded-lg transition-[background-color,color] duration-150 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30 focus-visible:ring-offset-2 focus-visible:ring-offset-background",
            active
              ? "bg-primary/10 text-primary"
              : "text-on-surface-variant hover:bg-control-hover hover:text-on-surface"
          )}
          type="button"
        >
          {active && (
            <span className="absolute left-0.5 top-1/2 h-3.5 w-0.5 -translate-y-1/2 rounded-full bg-primary" />
          )}
          <Icon className="h-4 w-4 stroke-[1.8]" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="right">
        <div className="space-y-0.5">
          <div>{label}</div>
          {description && (
            <div className="text-caption text-on-surface-variant">
              {description}
            </div>
          )}
        </div>
      </TooltipContent>
    </Tooltip>
  );
}

export function SideRail({ activePanel, onPanelChange }: SideRailProps) {
  const toggle = (label: string) =>
    onPanelChange(activePanel === label ? null : label);

  return (
    <aside
      aria-label="Workspace navigation"
      className="panel-raised pointer-events-auto flex flex-col items-center gap-1.5 rounded-2xl px-1 py-2"
    >
      <div
        className="flex flex-col gap-1"
        role="group"
        aria-label="Workspace panels"
      >
        {PRIMARY_ITEMS.map((item) => (
          <RailButton
            key={item.label}
            {...item}
            active={activePanel === item.label}
            onClick={() => toggle(item.label)}
          />
        ))}
      </div>

      <div className="h-px w-7 bg-divider" />

      <div className="flex flex-col gap-1" role="group" aria-label="System">
        <RailButton
          icon={Settings}
          label="Settings"
          description="Open application settings"
          active={activePanel === "Settings"}
          onClick={() => toggle("Settings")}
        />
      </div>
    </aside>
  );
}
