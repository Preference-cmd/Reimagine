import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowLeft, Redo2, Undo2 } from "lucide-react";
import { Link, useNavigate, useParams } from "@tanstack/react-router";
import * as m from "$paraglide/messages";
import { useProjectStore } from "@/store/project";
import { opInvoke, readAssetPreview, subscribeEvents, type AssetPreview } from "@/ipc";
import {
  CollectionDocumentSchema,
  type CollectionDocument,
  type CollectionSlot,
} from "@/ipc/schemas";

const PREVIEW_LINES = 4;

function SlotItem({ projectId, slot }: { projectId: string; slot: CollectionSlot }) {
  const [preview, setPreview] = useState<AssetPreview | null>(null);
  const [slotError, setSlotError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setPreview(null);
    setSlotError(null);
    readAssetPreview(projectId, slot.asset_id, PREVIEW_LINES).then(
      (result) => {
        if (active) setPreview(result);
      },
      (err: unknown) => {
        if (active) setSlotError(err instanceof Error ? err.message : String(err));
      },
    );
    return () => {
      active = false;
    };
  }, [projectId, slot.asset_id]);

  const mediaType = preview?.mediaType ?? "";

  return (
    <div className="flex flex-col rounded-xl border border-outline bg-surface p-4 transition-all">
      <div className="flex items-center justify-between border-b border-outline/50 pb-2">
        <span className="font-mono text-caption font-semibold text-primary">
          {m["collections.slotPrefix"]()} {slot.id}
        </span>
        <span className="rounded bg-control-hover px-2 py-0.5 font-mono text-caption text-on-surface-variant">
          {mediaType}
        </span>
      </div>

      <div className="mt-2 text-caption text-on-surface-variant">
        {m["collections.assetIdPrefix"]()} <span className="font-mono">{slot.asset_id}</span>
      </div>

      {slotError ? (
        <div className="mt-3 flex-1 rounded-lg bg-destructive/10 p-3 font-mono text-caption text-destructive">
          {slotError}
        </div>
      ) : (
        <div className="mt-3 flex-1 rounded-lg bg-surface-container-high/40 p-3 font-mono text-caption text-on-surface">
          {preview?.text ? (
            <pre className="whitespace-pre-wrap">{preview.text}</pre>
          ) : preview && mediaType.startsWith("image/") ? (
            <img src={preview.url} alt={slot.id} className="max-h-48 rounded object-contain" />
          ) : (
            <span className="italic text-on-surface-variant/60">
              {m["collections.noPreview"]()}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

export function CollectionView() {
  const { id } = useParams({ strict: false }) as { id: string };
  const activeProjectId = useProjectStore((s) => s.activeProjectId);
  const navigate = useNavigate();

  const [collection, setCollection] = useState<CollectionDocument | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // On project switch: reset state and navigate back to /collections
  const prevProjectIdRef = useRef(activeProjectId);
  useEffect(() => {
    if (prevProjectIdRef.current !== activeProjectId) {
      prevProjectIdRef.current = activeProjectId;
      setCollection(null);
      setError(null);
      setLoading(true);
      void navigate({ to: "/collections" });
    }
  }, [activeProjectId, navigate]);

  const fetchCollection = useCallback(async () => {
    if (!id) return;
    setLoading(true);
    try {
      const res = await opInvoke(activeProjectId, "collection.get", {
        collection_id: id,
      });
      const output = CollectionDocumentSchema.parse(res.output);
      setCollection(output);
      setError(null);
    } catch (e) {
      setCollection(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [activeProjectId, id]);

  useEffect(() => {
    void fetchCollection();
  }, [fetchCollection]);

  // Subscribe to collection.changed events for this collection and project
  useEffect(() => {
    const unsubscribe = subscribeEvents((event) => {
      if (
        event.type === "document" &&
        event.kind === "collection.changed" &&
        event.projectId === activeProjectId &&
        event.documentId === id
      ) {
        void fetchCollection();
      }
    });
    return () => unsubscribe();
  }, [activeProjectId, fetchCollection, id]);

  async function handleUndo() {
    if (!id) return;
    try {
      await opInvoke(activeProjectId, "collection.undo", { collection_id: id });
      await fetchCollection();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function handleRedo() {
    if (!id) return;
    try {
      await opInvoke(activeProjectId, "collection.redo", { collection_id: id });
      await fetchCollection();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="flex h-full flex-col bg-background p-6">
      {/* Top back navigation */}
      <div className="mb-4">
        <Link
          to="/collections"
          className="inline-flex items-center gap-1.5 text-caption font-medium text-on-surface-variant hover:text-on-surface"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          {m["collections.backToCollections"]()}
        </Link>
      </div>

      {/* Header */}
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-display-sm font-semibold text-on-surface">
            {collection?.name ?? m["collections.defaultTitle"]()}
          </h1>
          <p className="mt-1 text-body-sm text-on-surface-variant">
            {collection
              ? `${collection.slots.length} ${
                  collection.slots.length === 1 ? m["collections.slot"]() : m["collections.slots"]()
                } • ${m["collections.version"]()} ${collection.version}`
              : m["collections.loading"]()}
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => void handleUndo()}
            title={m["collections.undo"]()}
            className="flex h-9 cursor-pointer items-center gap-1.5 rounded-lg border border-outline bg-surface px-3 text-body-sm font-medium text-on-surface transition-colors hover:bg-control-hover"
          >
            <Undo2 className="h-4 w-4" />
            {m["collections.undo"]()}
          </button>
          <button
            type="button"
            onClick={() => void handleRedo()}
            title={m["collections.redo"]()}
            className="flex h-9 cursor-pointer items-center gap-1.5 rounded-lg border border-outline bg-surface px-3 text-body-sm font-medium text-on-surface transition-colors hover:bg-control-hover"
          >
            <Redo2 className="h-4 w-4" />
            {m["collections.redo"]()}
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 rounded-lg bg-destructive/10 p-3 text-body-sm text-destructive">
          {error}
        </div>
      )}

      {loading ? (
        <div className="flex flex-1 items-center justify-center text-body-sm text-on-surface-variant">
          {m["collections.loadingCollection"]()}
        </div>
      ) : !collection ? (
        <div className="flex flex-1 items-center justify-center text-body-sm text-on-surface-variant">
          {m["collections.notFound"]()}
        </div>
      ) : collection.slots.length === 0 ? (
        <div className="flex flex-1 items-center justify-center text-body-sm text-on-surface-variant">
          {m["collections.emptySlots"]()}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {collection.slots.map((slot) => (
            <SlotItem key={slot.id} projectId={activeProjectId} slot={slot} />
          ))}
        </div>
      )}
    </div>
  );
}
