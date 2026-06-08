import { useState } from "react";
import { Search, Folder, FolderOpen, Minus, X, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

type FolderItem = {
  name: string;
  count?: number;
  depth: 0 | 1 | 2;
  open?: boolean;
};

const FOLDERS: FolderItem[] = [
  { name: "Record", depth: 0 },
  { name: "Data Fields", depth: 0 },
  { name: "Documents", depth: 0, count: 2 },
  { name: "Fee", depth: 0, count: 4 },
  { name: "Workflow", depth: 0, open: true },
  { name: "BUSA-2026-01", depth: 1, open: true },
  { name: "STD_DEMOv2", depth: 2, count: 4 },
  { name: "License", depth: 2, count: 3 },
  { name: "DUKQ-2025-12", depth: 1 },
  { name: "Workflow History", depth: 0, count: 1 },
  { name: "Inspections", depth: 0 },
];

export function ExplorerPanel() {
  const [tab, setTab] = useState<"Folders" | "Tags">("Folders");

  return (
    <div className="panel-glass absolute left-4 top-[72px] bottom-4 z-30 flex w-72 flex-col rounded-2xl shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between p-5">
        <span className="text-body-md font-semibold text-zinc-200">Explorer</span>
        <div className="flex gap-3 text-zinc-600">
          <Minus className="h-4 w-4 cursor-pointer" />
          <X className="h-4 w-4 cursor-pointer" />
        </div>
      </div>

      {/* Search */}
      <div className="px-4 pb-4">
        <div className="relative">
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-zinc-500" />
          <input
            className="w-full rounded-xl border-none bg-zinc-900 py-2 pl-10 text-body-sm text-zinc-300 placeholder-zinc-600 focus:ring-1 focus:ring-white/10"
            placeholder="Search"
            type="text"
          />
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 px-4 pb-4">
        {(["Folders", "Tags"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={cn(
              "flex-1 rounded-lg py-1.5 text-label-caps transition-colors",
              t === tab
                ? "border border-white/5 bg-zinc-800 text-white shadow-inner"
                : "text-zinc-500 hover:text-zinc-300",
            )}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Tree */}
      <div className="scrollbar-hide flex-1 overflow-y-auto px-4 pb-6 text-body-md text-zinc-400">
        <ul className="space-y-3">
          {FOLDERS.map((f, i) => (
            <li
              key={i}
              className="flex items-center justify-between"
              style={{ paddingLeft: f.depth * 24 }}
            >
              <div
                className={cn(
                  "flex items-center gap-3",
                  f.open ? "font-medium text-zinc-200" : "opacity-60",
                )}
              >
                {f.open && (
                  <ChevronRight className="h-4 w-4 rotate-90" />
                )}
                {f.open ? (
                  <FolderOpen className="h-4 w-4" />
                ) : (
                  <Folder className="h-4 w-4" />
                )}
                <span>{f.name}</span>
              </div>
              {f.count !== undefined && (
                <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-label-caps">
                  {f.count}
                </span>
              )}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}