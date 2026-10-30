import { useState } from "react";
import { Boxes, Download, HardDrive, Search } from "lucide-react";
import { cn } from "@/lib/utils";
import { useModels } from "@/hooks/queries";
import { ICON_STROKE } from "@/design/tokens";
import { ModelDownloadDialog } from "./ModelDownloadDialog";

function formatBytes(bytes: number | null): string {
  if (bytes == null || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex++;
  }
  return `${value.toFixed(2)} ${units[unitIndex]}`;
}

/**
 * ModelsView — full-page model browser.
 * Shows installed models with search, and an install button.
 */
export function ModelsView() {
  const { data: models = [], isLoading: loading } = useModels();
  const [query, setQuery] = useState("");
  const [downloadOpen, setDownloadOpen] = useState(false);

  const filtered = models.filter(
    (m) =>
      !query ||
      m.displayName.toLowerCase().includes(query.toLowerCase()) ||
      m.modelSeries?.toLowerCase().includes(query.toLowerCase()),
  );

  return (
    <div className="flex h-full flex-col overflow-y-auto bg-ground p-6">
      {/* Header */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-display-sm font-semibold text-fg">Models</h1>
          <p className="mt-1 text-body-sm text-fg3">Local model index</p>
        </div>
        <button
          type="button"
          onClick={() => setDownloadOpen(true)}
          className="flex h-9 cursor-pointer items-center gap-2 rounded-lg bg-accent px-4 text-body-sm font-medium text-on-accent transition-colors hover:bg-accent-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/30"
        >
          <Download className="h-4 w-4" strokeWidth={ICON_STROKE} />
          Install model
        </button>
      </div>

      {/* Search */}
      <div className="relative mb-4">
        <Search
          className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-fg3"
          strokeWidth={ICON_STROKE}
        />
        <input
          className="h-10 w-full rounded-lg border border-hair bg-surface px-4 pl-10 text-body-sm text-fg outline-none transition-[border-color,box-shadow] placeholder:text-fg3 focus:border-accent-line focus:ring-2 focus:ring-accent/20"
          placeholder="Search models…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>

      {/* Model list */}
      <div className="min-h-0 flex-1 overflow-y-auto scrollbar-hide">
        {loading ? (
          <div className="flex items-center justify-center py-12 text-body-sm text-fg3">
            Indexing models…
          </div>
        ) : filtered.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-fg3">
            <HardDrive className="mb-3 h-10 w-10 opacity-30" strokeWidth={ICON_STROKE} />
            <p className="text-body-sm font-medium">No models found</p>
            <p className="mt-1 text-caption">
              {query ? "Try a different search" : "Install a model to get started"}
            </p>
          </div>
        ) : (
          <div className="grid gap-2">
            {filtered.map((model) => (
              <div
                key={model.id}
                className={cn(
                  "flex items-center gap-3 rounded-lg border border-hair bg-surface p-3 shadow-1 transition-colors hover:bg-hover",
                )}
              >
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-accent-tint text-accent">
                  <Boxes className="h-5 w-5" strokeWidth={ICON_STROKE} />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="truncate text-body-sm font-medium text-fg">
                    {model.displayName}
                  </div>
                  <div className="truncate text-caption text-fg3">
                    {model.modelSeries} · {formatBytes(model.sizeBytes)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <ModelDownloadDialog
        open={downloadOpen}
        initialRepoId={null}
        onClose={() => setDownloadOpen(false)}
        onInstalled={() => {}}
      />
    </div>
  );
}
