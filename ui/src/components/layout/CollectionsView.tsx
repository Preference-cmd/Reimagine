import { useState } from "react";
import { ArrowRight, FolderKanban, Layers, Upload } from "lucide-react";
import { Link, useNavigate } from "@tanstack/react-router";
import * as m from "$paraglide/messages";
import { useBoardStore } from "@/store/board";
import { useProjectStore } from "@/store/project";
import { opInvoke } from "@/ipc";

export async function pickFiles(): Promise<string[] | null> {
  try {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const res = await open({
      multiple: true,
      directory: false,
    });
    if (!res) return null;
    return Array.isArray(res) ? res : [res];
  } catch {
    if (
      typeof window !== "undefined" &&
      (window as unknown as { __MOCK_IMPORT_PATHS__?: string[] }).__MOCK_IMPORT_PATHS__
    ) {
      return (window as unknown as { __MOCK_IMPORT_PATHS__: string[] }).__MOCK_IMPORT_PATHS__;
    }
    return null;
  }
}

export function CollectionsView() {
  const activeProjectId = useProjectStore((s) => s.activeProjectId);
  const items = useBoardStore((s) => s.items);
  const navigate = useNavigate();
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const collectionItems = items.filter(
    (item) =>
      typeof item.kind === "object" &&
      item.kind !== null &&
      "type" in item.kind &&
      item.kind.type === "collection",
  );

  async function handleImport() {
    setError(null);
    try {
      const paths = await pickFiles();
      if (!paths || paths.length === 0) return;

      setImporting(true);
      const importRes = await opInvoke(activeProjectId, "asset.import", { paths });
      const assets = (importRes.output as { assets?: string[] })?.assets ?? [];
      if (assets.length === 0) {
        throw new Error("No assets were imported");
      }

      const createRes = await opInvoke(activeProjectId, "collection.create", {
        name: `${m["collections.collectionPrefix"]()} ${collectionItems.length + 1}`,
        assets,
      });

      const colId = (createRes.output as { collection_id?: string })?.collection_id;
      if (colId) {
        await navigate({ to: `/collections/${colId}` as string });
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setImporting(false);
    }
  }

  return (
    <div className="flex h-full flex-col bg-background p-6">
      {/* Header */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-display-sm font-semibold text-on-surface">
            {m["collections.title"]()}
          </h1>
          <p className="mt-1 text-body-sm text-on-surface-variant">{m["collections.subtitle"]()}</p>
        </div>
        <button
          type="button"
          onClick={() => void handleImport()}
          disabled={importing}
          className="flex h-9 cursor-pointer items-center gap-2 rounded-lg bg-primary px-4 text-body-sm font-medium text-on-primary transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30 disabled:opacity-50"
        >
          <Upload className="h-4 w-4" />
          {importing ? m["collections.importing"]() : m["collections.import"]()}
        </button>
      </div>

      {error && (
        <div className="mb-4 rounded-lg bg-destructive/10 p-3 text-body-sm text-destructive">
          {error}
        </div>
      )}

      {/* Collection items list or empty state */}
      {collectionItems.length === 0 ? (
        <div className="flex min-h-0 flex-1 flex-col items-center justify-center text-on-surface-variant">
          <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-control-hover">
            <FolderKanban className="h-8 w-8 opacity-40" />
          </div>
          <p className="text-body-sm font-medium">{m["collections.emptyTitle"]()}</p>
          <p className="mt-1 max-w-xs text-center text-caption">{m["collections.emptyDesc"]()}</p>
          <button
            type="button"
            onClick={() => void handleImport()}
            className="mt-6 flex h-9 cursor-pointer items-center gap-2 rounded-lg bg-control-hover px-4 text-body-sm font-medium text-on-surface transition-colors hover:bg-control-hover/80"
          >
            <Upload className="h-4 w-4" />
            {m["collections.importFiles"]()}
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {collectionItems.map((item) => {
            const colId = (item.kind as { collection_id: string }).collection_id;
            return (
              <Link
                key={item.id}
                to={`/collections/${colId}` as string}
                className="group flex flex-col justify-between rounded-xl border border-outline bg-surface p-4 transition-all hover:border-primary/40 hover:shadow-sm"
              >
                <div>
                  <div className="flex items-center justify-between">
                    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 text-primary">
                      <Layers className="h-4 w-4" />
                    </div>
                    <ArrowRight className="h-4 w-4 text-on-surface-variant transition-transform group-hover:translate-x-0.5 group-hover:text-primary" />
                  </div>
                  <h3 className="mt-3 text-body-base font-medium text-on-surface">
                    {m["collections.collectionPrefix"]()} {colId}
                  </h3>
                  <p className="mt-1 text-caption text-on-surface-variant">
                    {m["collections.boardItemPrefix"]()} {item.id}
                  </p>
                </div>
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}
