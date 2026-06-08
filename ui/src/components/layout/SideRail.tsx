import {
  Home,
  Layers,
  Workflow,
  Calendar,
  FileText,
  Settings,
  LogOut,
} from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

type RailItem = {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  active?: boolean;
};

const PRIMARY_ITEMS: RailItem[] = [
  { icon: Home, label: "Home" },
  { icon: Layers, label: "Layers" },
  { icon: Workflow, label: "Workflow", active: true },
  { icon: Calendar, label: "Calendar" },
  { icon: FileText, label: "Documents" },
  { icon: Settings, label: "Settings" },
];

const SECONDARY_ITEMS: RailItem[] = [
  { icon: LogOut, label: "Log out" },
];

function RailButton({ icon: Icon, label, active }: RailItem) {
  return (
    <Tooltip>
      <TooltipTrigger>
        <button
          className={
            active
              ? "text-green-500"
              : "text-zinc-600 transition-colors hover:text-white"
          }
        >
          <Icon className="h-5 w-5" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="right">{label}</TooltipContent>
    </Tooltip>
  );
}

export function SideRail() {
  return (
    <aside className="absolute left-0 top-14 bottom-0 z-40 flex w-16 flex-col items-center border-r border-white/5 bg-black py-6">
      <div className="flex flex-col gap-8">
        {PRIMARY_ITEMS.map((item) => (
          <RailButton key={item.label} {...item} />
        ))}
      </div>
      <div className="mt-auto flex flex-col gap-6">
        {SECONDARY_ITEMS.map((item) => (
          <RailButton key={item.label} {...item} />
        ))}
      </div>
    </aside>
  );
}