import { useEffect, useRef, useState } from "react";
import { ArrowLeft, Boxes, CheckCircle2, Download, Loader2, Search, X } from "lucide-react";
import { Dialog } from "radix-ui";
import { Button } from "@/components/ui/button";
import { downloadHuggingfaceModel, getModelCard, searchModels } from "@/ipc";
import type {
  DownloadEventPayload,
  DownloadHuggingfaceModelArgs,
  ModelCard,
  ModelCatalogEntry,
  ModelDownloadOutput,
} from "@/ipc";

const SEARCH_DELAY_MS = 300;

function formatBytes(bytes: number): string {
  if (bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex++;
  }
  return `${value.toFixed(2)} ${units[unitIndex]}`;
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

type Phase = "browse" | "card" | "downloading" | "done";

type DownloadProgress = {
  bytes: number;
  totalBytes: number | null;
  message: string | null;
};

type ModelDownloadDialogProps = {
  open: boolean;
  initialRepoId: string | null;
  onClose: () => void;
  onInstalled: () => void;
};

const PHASE_TITLES: Record<Phase, string> = {
  browse: "Install model",
  card: "Model card",
  downloading: "Downloading",
  done: "Installed",
};

export function ModelDownloadDialog({
  open,
  initialRepoId,
  onClose,
  onInstalled,
}: ModelDownloadDialogProps) {
  const [phase, setPhase] = useState<Phase>("browse");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<ModelCatalogEntry[]>([]);
  const [searching, setSearching] = useState(false);
  const [selectedRepoId, setSelectedRepoId] = useState<string | null>(null);
  const [card, setCard] = useState<ModelCard | null>(null);
  const [cardError, setCardError] = useState<string | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [progress, setProgress] = useState<DownloadProgress>({
    bytes: 0,
    totalBytes: null,
    message: null,
  });
  const [output, setOutput] = useState<ModelDownloadOutput | null>(null);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  useEffect(() => {
    if (!open) return;
    setQuery("");
    setResults([]);
    setSearching(false);
    setDownloadError(null);
    setOutput(null);
    setProgress({ bytes: 0, totalBytes: null, message: null });
    if (initialRepoId) {
      void openRepo(initialRepoId);
    } else {
      setSelectedRepoId(null);
      setCard(null);
      setCardError(null);
      setPhase("browse");
    }
  }, [open, initialRepoId]);

  useEffect(() => {
    if (!open || phase !== "browse") return;
    const needle = query.trim();
    if (!needle) {
      setResults([]);
      setSearching(false);
      return;
    }
    setSearching(true);
    let cancelled = false;
    const timer = setTimeout(() => {
      searchModels(needle)
        .then((entries) => {
          if (!cancelled) setResults(entries);
        })
        .catch(() => {
          if (!cancelled) setResults([]);
        })
        .finally(() => {
          if (!cancelled) setSearching(false);
        });
    }, SEARCH_DELAY_MS);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [open, phase, query]);

  async function openRepo(repoId: string) {
    setSelectedRepoId(repoId);
    setCard(null);
    setCardError(null);
    setDownloadError(null);
    setOutput(null);
    setPhase("card");
    try {
      const fetched = await getModelCard(repoId);
      if (alive.current) setCard(fetched);
    } catch (error) {
      if (alive.current) setCardError(errorMessage(error));
    }
  }

  function applyDownloadEvent(event: DownloadEventPayload) {
    setProgress((prev) => {
      if (event.status === "started") {
        return { bytes: 0, totalBytes: event.totalBytes ?? null, message: null };
      }
      if (event.status === "in_progress") {
        return {
          ...prev,
          bytes: prev.bytes + event.bytesDownloaded,
          message: event.message ?? prev.message,
        };
      }
      if (event.status === "completed") {
        return {
          bytes: event.totalBytes ?? prev.bytes,
          totalBytes: event.totalBytes ?? prev.totalBytes,
          message: null,
        };
      }
      return prev;
    });
  }

  async function startDownload() {
    if (!selectedRepoId) return;
    setDownloadError(null);
    setProgress({ bytes: 0, totalBytes: null, message: null });
    setPhase("downloading");
    const args: DownloadHuggingfaceModelArgs = {
      repoId: selectedRepoId,
      revision: "main",
      targetRelativeDir: selectedRepoId,
      fromCatalog: true,
    };
    try {
      const result = await downloadHuggingfaceModel(args, (event) => {
        applyDownloadEvent(event);
      });
      if (!alive.current) return;
      setOutput(result);
      setPhase("done");
      onInstalled();
    } catch (error) {
      if (!alive.current) return;
      setDownloadError(errorMessage(error));
      setPhase("card");
    }
  }

  const fraction = progress.totalBytes ? Math.min(1, progress.bytes / progress.totalBytes) : 0;

  return (
    <Dialog.Root open={open} onOpenChange={(nextOpen) => !nextOpen && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-[var(--overlay-z-modal-backdrop)] bg-background/18 backdrop-blur-[2px]" />
        <Dialog.Content
          aria-label="Install model"
          className="fixed left-1/2 top-1/2 z-[var(--overlay-z-modal)] flex max-h-[min(560px,calc(100vh-48px))] w-[min(420px,calc(100vw-48px))] -translate-x-1/2 -translate-y-1/2 flex-col overflow-hidden rounded-xl border border-outline bg-surface shadow-modal outline-none"
        >
          <header className="flex h-12 shrink-0 items-center justify-between border-b border-outline px-4">
            <div className="flex min-w-0 items-center gap-2">
              {phase === "card" && (
                <button
                  type="button"
                  aria-label="Back to search"
                  onClick={() => setPhase("browse")}
                  className="flex h-6 w-6 shrink-0 cursor-pointer items-center justify-center rounded-md text-on-surface-variant transition-colors hover:bg-control-hover hover:text-on-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30"
                >
                  <ArrowLeft className="h-4 w-4" />
                </button>
              )}
              <Dialog.Title className="truncate text-body-sm font-semibold text-on-surface">
                {PHASE_TITLES[phase]}
              </Dialog.Title>
            </div>
            <Dialog.Close asChild>
              <button
                type="button"
                aria-label="Close install dialog"
                className="flex h-6 w-6 shrink-0 cursor-pointer items-center justify-center rounded-md text-on-surface-variant transition-colors hover:bg-control-hover hover:text-on-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30"
              >
                <X className="h-4 w-4" />
              </button>
            </Dialog.Close>
          </header>

          {phase === "browse" && (
            <>
              <div className="p-4">
                <label className="relative block">
                  <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-on-surface-variant" />
                  <input
                    className="h-8 w-full rounded-full border border-outline bg-surface-container-low px-3 pl-8 text-caption text-on-surface outline-none transition-[border-color,box-shadow] placeholder:text-on-surface-variant focus:border-primary/30 focus:ring-2 focus:ring-primary/10"
                    placeholder="Search HuggingFace models"
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    type="text"
                    autoFocus
                  />
                </label>
              </div>
              <div className="scrollbar-hide flex-1 overflow-y-auto px-2 pb-3">
                {searching ? (
                  <div className="mx-1 flex items-center gap-2 rounded-lg bg-control-hover/60 px-3 py-2 text-caption text-on-surface-variant">
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    Searching hub...
                  </div>
                ) : results.length === 0 ? (
                  <div className="mx-1 mt-2 rounded-lg bg-control-hover/60 px-3 py-2 text-caption text-on-surface-variant">
                    {query.trim()
                      ? "No models found."
                      : "Search HuggingFace or pick a model from the explorer."}
                  </div>
                ) : (
                  <ul className="space-y-0.5">
                    {results.map((entry) => (
                      <li key={entry.id}>
                        <button
                          type="button"
                          onClick={() => void openRepo(entry.id)}
                          className="grid w-full cursor-pointer grid-cols-[auto_minmax(0,1fr)] gap-2 rounded-lg px-2 py-1.5 text-left transition-colors hover:bg-control-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/20"
                        >
                          <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-control-hover text-on-surface-variant">
                            <Boxes className="h-3.5 w-3.5" />
                          </span>
                          <span className="min-w-0 flex-1">
                            <span className="block truncate text-caption font-medium text-on-surface">
                              {entry.id}
                            </span>
                            <span className="mt-0.5 block truncate text-caption text-on-surface-variant">
                              {entry.pipelineTag ?? "model"} / {entry.downloads.toLocaleString()}{" "}
                              downloads
                            </span>
                          </span>
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </>
          )}

          {phase === "card" && (
            <>
              <div className="scrollbar-hide flex-1 overflow-y-auto p-4">
                {cardError ? (
                  <div className="rounded-lg bg-error-container/40 px-3 py-2 text-caption text-on-surface">
                    {cardError}
                  </div>
                ) : !card ? (
                  <div className="flex items-center gap-2 text-caption text-on-surface-variant">
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    Loading model card...
                  </div>
                ) : (
                  <>
                    <div className="truncate text-body-sm font-semibold text-on-surface">
                      {card.entry.id}
                    </div>
                    <div className="mt-0.5 truncate text-caption text-on-surface-variant">
                      {card.entry.author ?? "Unknown author"}
                    </div>

                    <div className="mt-3 grid grid-cols-2 gap-1">
                      <DialogSummaryCell label="Format" value={card.detectedFormat} />
                      <DialogSummaryCell
                        label="Estimated size"
                        value={formatBytes(card.estimatedDownloadSize)}
                      />
                      <DialogSummaryCell label="Files" value={String(card.fileCount)} />
                      <DialogSummaryCell
                        label="Components"
                        value={card.components.length ? card.components.join(", ") : "Unknown"}
                      />
                    </div>

                    {card.modelSummary && (
                      <p className="mt-3 text-caption text-on-surface-variant">
                        {card.modelSummary}
                      </p>
                    )}

                    {downloadError && (
                      <div className="mt-3 rounded-lg bg-error-container/40 px-3 py-2 text-caption text-on-surface">
                        {downloadError}
                      </div>
                    )}
                  </>
                )}
              </div>
              <footer className="flex shrink-0 items-center justify-end gap-2 border-t border-outline px-4 py-3">
                <Button size="sm" onClick={() => void startDownload()} disabled={!card}>
                  <Download className="h-3.5 w-3.5" />
                  Download
                </Button>
              </footer>
            </>
          )}

          {phase === "downloading" && (
            <div className="scrollbar-hide flex-1 overflow-y-auto p-4">
              <div className="truncate text-body-sm font-semibold text-on-surface">
                {selectedRepoId}
              </div>
              <div className="mt-0.5 truncate text-caption text-on-surface-variant">
                {progress.message ?? "Downloading model files"}
              </div>
              <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-surface-container-high">
                <div
                  className="h-full origin-left rounded-full bg-primary transition-transform duration-200 ease-out motion-reduce:transition-none"
                  style={{ transform: `scaleX(${Math.max(0.04, fraction)})` }}
                />
              </div>
              <div className="mt-2 flex items-center justify-between gap-3 text-caption text-on-surface-variant">
                <span className="truncate">{formatBytes(progress.bytes)} transferred</span>
                {progress.totalBytes ? (
                  <span className="shrink-0">{formatBytes(progress.totalBytes)} total</span>
                ) : (
                  <span className="shrink-0">Estimating…</span>
                )}
              </div>
            </div>
          )}

          {phase === "done" && (
            <div className="flex flex-1 flex-col items-center justify-center gap-2 p-6 text-center">
              <CheckCircle2 className="h-8 w-8 text-status-ready" />
              <div className="text-body-sm font-semibold text-on-surface">Model installed</div>
              <div className="max-w-full truncate text-caption text-on-surface-variant">
                {output
                  ? `${output.repoId} · ${formatBytes(output.totalBytes)} · ${
                      output.detectedFormat ?? output.provider
                    }`
                  : selectedRepoId}
              </div>
              <Button size="sm" className="mt-3" onClick={onClose}>
                Done
              </Button>
            </div>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function DialogSummaryCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded-lg bg-control-hover/60 px-2 py-1.5">
      <div className="truncate text-caption text-on-surface-variant">{label}</div>
      <div className="truncate text-caption font-semibold text-on-surface">{value}</div>
    </div>
  );
}
