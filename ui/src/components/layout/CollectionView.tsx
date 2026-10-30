import { useEffect, useRef, useState } from "react";
import { ArrowLeft, CircleAlert, Redo2, Undo2 } from "lucide-react";
import { useNavigate, useParams, useRouter } from "@tanstack/react-router";
import * as m from "$paraglide/messages";
import { ICON_STROKE } from "@/design/tokens";
import { useProjectStore } from "@/store/project";
import { opInvoke, readAssetPreview, type AssetPreview } from "@/ipc";
import { type CollectionSlot, type SlotRunState } from "@/ipc/schemas";
import { useCollection } from "@/hooks/useCollection";
import { useCollectionRun } from "@/hooks/useCollectionRun";

const PREVIEW_LINES = 4;

function slotStateLabel(state: SlotRunState): string {
  switch (state) {
    case "queued":
      return m["collections.slotState.queued"]();
    case "running":
      return m["collections.slotState.running"]();
    case "succeeded":
      return m["collections.slotState.succeeded"]();
    case "failed":
      return m["collections.slotState.failed"]();
    case "stale":
      return m["collections.slotState.stale"]();
    case "cancelled":
      return m["collections.slotState.cancelled"]();
    case "needs_confirmation":
      return m["collections.slotState.needsConfirmation"]();
  }
}

function slotStateClass(state: SlotRunState): string {
  switch (state) {
    case "running":
      return "bg-info/10 text-fg";
    case "succeeded":
      return "bg-ok/10 text-fg";
    case "failed":
      return "bg-err/10 text-fg";
    case "needs_confirmation":
      return "bg-warn/10 text-fg";
    default:
      return "bg-tile text-fg2";
  }
}

/** The tinted pill keeps the text at `fg`; the dot alone carries the state colour. */
function slotStateDotClass(state: SlotRunState): string | null {
  switch (state) {
    case "running":
      return "bg-info";
    case "succeeded":
      return "bg-ok";
    case "failed":
      return "bg-err";
    case "needs_confirmation":
      return "bg-warn";
    default:
      return null;
  }
}

function SlotItem({
  projectId,
  slot,
  state,
}: {
  projectId: string;
  slot: CollectionSlot;
  state?: SlotRunState;
}) {
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
    <div className="flex flex-col rounded-lg border border-hair bg-surface p-4 shadow-1 transition-all">
      <div className="flex items-center justify-between border-b border-hair pb-2">
        <span className="font-mono text-caption font-semibold text-accent">
          {m["collections.slotPrefix"]()} {slot.id}
        </span>
        <span className="flex items-center gap-2">
          {state && (
            <span
              className={`flex items-center gap-1.5 rounded px-2 py-0.5 font-mono text-caption ${slotStateClass(state)}`}
            >
              {slotStateDotClass(state) && (
                <span
                  aria-hidden="true"
                  className={`size-1.5 shrink-0 rounded-full ${slotStateDotClass(state)}`}
                />
              )}
              {slotStateLabel(state)}
            </span>
          )}
          <span className="rounded bg-tile px-2 py-0.5 font-mono text-caption text-fg2">
            {mediaType}
          </span>
        </span>
      </div>

      <div className="mt-2 text-caption text-fg2">
        {m["collections.assetIdPrefix"]()} <span className="font-mono">{slot.asset_id}</span>
      </div>

      {slotError ? (
        <div className="mt-3 flex-1 rounded-lg border-l-2 border-err bg-err/10 p-3 font-mono text-caption text-fg">
          {slotError}
        </div>
      ) : (
        <div className="mt-3 flex-1 rounded-lg bg-tile p-3 font-mono text-caption text-fg">
          {preview?.text ? (
            <pre className="whitespace-pre-wrap">{preview.text}</pre>
          ) : preview && mediaType.startsWith("image/") ? (
            <img src={preview.url} alt={slot.id} className="max-h-48 rounded object-contain" />
          ) : (
            <span className="italic text-fg2">{m["collections.noPreview"]()}</span>
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
  const router = useRouter();

  const {
    collection,
    loading,
    error: collectionError,
    refresh: fetchCollection,
  } = useCollection(activeProjectId, id);
  const { slotStates } = useCollectionRun(activeProjectId, id);
  const [actionError, setActionError] = useState<string | null>(null);
  const error = collectionError ?? actionError;

  // On project switch: reset state and navigate back to the library
  const prevProjectIdRef = useRef(activeProjectId);
  useEffect(() => {
    if (prevProjectIdRef.current !== activeProjectId) {
      prevProjectIdRef.current = activeProjectId;
      setActionError(null);
      void navigate({ to: "/library" });
    }
  }, [activeProjectId, navigate]);

  async function handleUndo() {
    if (!id) return;
    try {
      setActionError(null);
      await opInvoke(activeProjectId, "collection.undo", { collection_id: id });
      await fetchCollection();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    }
  }

  async function handleRedo() {
    if (!id) return;
    try {
      setActionError(null);
      await opInvoke(activeProjectId, "collection.redo", { collection_id: id });
      await fetchCollection();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    }
  }

  // Return to whichever mode opened the detail; a direct link has no history
  // entry, so it falls back to the library.
  function handleBack() {
    if (router.history.canGoBack()) {
      router.history.back();
      return;
    }
    void navigate({ to: "/library" });
  }

  return (
    <div className="flex h-full flex-col overflow-y-auto bg-ground p-6">
      {/* Top back navigation */}
      <div className="mb-4">
        <button
          type="button"
          onClick={handleBack}
          className="inline-flex cursor-pointer items-center gap-1.5 rounded-sm text-caption font-medium text-fg2 transition-colors hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/30"
        >
          <ArrowLeft className="h-3.5 w-3.5" strokeWidth={ICON_STROKE} />
          {m["common.back"]()}
        </button>
      </div>

      {/* Header */}
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-display-sm font-semibold text-fg">
            {collection?.name ?? m["collections.defaultTitle"]()}
          </h1>
          <p className="mt-1 text-body-sm text-fg2">
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
            className="flex h-9 cursor-pointer items-center gap-1.5 rounded-lg border border-hair bg-surface px-3 text-body-sm font-medium text-fg transition-colors hover:bg-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/30"
          >
            <Undo2 className="h-4 w-4" strokeWidth={ICON_STROKE} />
            {m["collections.undo"]()}
          </button>
          <button
            type="button"
            onClick={() => void handleRedo()}
            title={m["collections.redo"]()}
            className="flex h-9 cursor-pointer items-center gap-1.5 rounded-lg border border-hair bg-surface px-3 text-body-sm font-medium text-fg transition-colors hover:bg-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/30"
          >
            <Redo2 className="h-4 w-4" strokeWidth={ICON_STROKE} />
            {m["collections.redo"]()}
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 flex items-center gap-2 rounded-lg border-l-2 border-err bg-err/10 p-3 text-body-sm text-fg">
          <CircleAlert
            className="size-4 shrink-0 text-err"
            aria-hidden="true"
            strokeWidth={ICON_STROKE}
          />
          {error}
        </div>
      )}

      {loading ? (
        <div className="flex flex-1 items-center justify-center text-body-sm text-fg2">
          {m["collections.loadingCollection"]()}
        </div>
      ) : !collection ? (
        <div className="flex flex-1 items-center justify-center text-body-sm text-fg2">
          {m["collections.notFound"]()}
        </div>
      ) : collection.slots.length === 0 ? (
        <div className="flex flex-1 items-center justify-center text-body-sm text-fg2">
          {m["collections.emptySlots"]()}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {collection.slots.map((slot) => (
            <SlotItem
              key={slot.id}
              projectId={activeProjectId}
              slot={slot}
              state={slotStates[slot.id]}
            />
          ))}
        </div>
      )}
    </div>
  );
}
