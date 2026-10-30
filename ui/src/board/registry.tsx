import { useEffect, useRef, useState } from "react";
import { Boxes, FileImage, Layers, Play, Lock } from "lucide-react";
import { useNavigate } from "@tanstack/react-router";
import * as m from "$paraglide/messages";
import { useProjectStore } from "@/store/project";
import { useCollection } from "@/hooks/useCollection";
import type { BoardItemKind } from "@/ipc/schemas";

export type BoardCardProps = {
  id: string;
  kind: BoardItemKind | { type: string; [key: string]: unknown };
  size: { width: number; height: number };
  locked: boolean;
  selected?: boolean;
  isEditing?: boolean;
  onStartEdit?: () => void;
  onEndEdit?: () => void;
  onUpdateNote?: (content: string) => void;
};

export function NoteCard({
  kind,
  locked,
  isEditing,
  onStartEdit,
  onEndEdit,
  onUpdateNote,
}: BoardCardProps) {
  const content = (kind as { type: "note"; content: string }).content ?? "";
  const [draft, setDraft] = useState(content);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const sessionEndedRef = useRef(false);

  useEffect(() => {
    setDraft(content);
  }, [content]);

  const isEditingActive = Boolean(isEditing && !locked);

  useEffect(() => {
    if (isEditingActive) {
      sessionEndedRef.current = false;
      setDraft(content);
      const timer = setTimeout(() => {
        if (textareaRef.current) {
          textareaRef.current.focus();
          textareaRef.current.selectionStart = textareaRef.current.value.length;
          textareaRef.current.selectionEnd = textareaRef.current.value.length;
        }
      }, 0);
      return () => clearTimeout(timer);
    }
  }, [isEditingActive, content]);

  function handleBlur() {
    if (!isEditingActive || sessionEndedRef.current) return;
    sessionEndedRef.current = true;
    if (draft !== content) {
      onUpdateNote?.(draft);
    }
    onEndEdit?.();
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      sessionEndedRef.current = true;
      setDraft(content);
      onEndEdit?.();
      return;
    }
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      e.stopPropagation();
      if (sessionEndedRef.current) return;
      sessionEndedRef.current = true;
      if (draft !== content) {
        onUpdateNote?.(draft);
      }
      onEndEdit?.();
    }
  }

  return (
    <div
      className="flex h-full w-full flex-col overflow-hidden rounded-xl border border-outline bg-surface-container-high p-4 text-on-surface shadow-xs transition-shadow select-none hover:shadow-md"
      onDoubleClick={(e) => {
        e.stopPropagation();
        if (locked) return;
        onStartEdit?.();
      }}
      data-testid="note-card"
    >
      <div className="mb-2 flex items-center justify-between text-caption font-medium text-on-surface-variant">
        <span>{m["board.note"]()}</span>
        {locked && (
          <span
            title={m["board.locked"]()}
            aria-label={m["board.locked"]()}
            data-testid="lock-indicator"
          >
            <Lock className="h-3.5 w-3.5 text-on-surface-variant" aria-hidden="true" />
          </span>
        )}
      </div>
      {isEditingActive ? (
        <textarea
          ref={textareaRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={handleBlur}
          onKeyDown={handleKeyDown}
          className="nodrag nopan nowheel h-full w-full resize-none border-0 bg-transparent p-0 font-sans text-body-sm text-on-surface outline-none focus:ring-0"
          placeholder={m["board.notePlaceholder"]()}
          data-testid="note-textarea"
        />
      ) : (
        <div className="h-full w-full overflow-y-auto whitespace-pre-wrap font-sans text-body-sm leading-relaxed text-on-surface">
          {content ? (
            content
          ) : (
            <span className="italic text-on-surface-variant/60">
              {m["board.doubleClickToEdit"]()}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

export function CollectionCard({ kind, locked }: BoardCardProps) {
  const collectionId = (kind as { type: "collection"; collection_id: string }).collection_id;
  const activeProjectId = useProjectStore((s) => s.activeProjectId);
  const navigate = useNavigate();
  const { collection, loading } = useCollection(activeProjectId, collectionId);

  const slotCount = collection ? collection.slots.length : 0;
  const slotLabel = slotCount === 1 ? m["collections.slot"]() : m["collections.slots"]();

  return (
    <div
      onClick={() => void navigate({ to: "/collections/$id", params: { id: collectionId } })}
      className="flex h-full w-full cursor-pointer flex-col justify-between rounded-xl border border-outline bg-surface p-4 text-on-surface shadow-xs transition-all select-none hover:border-primary/50 hover:bg-surface-container-high hover:shadow-md"
      data-testid="collection-card"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <Layers className="h-4 w-4" />
          </div>
          <div className="min-w-0 flex-1">
            <h3 className="truncate font-sans text-body-sm font-semibold text-on-surface">
              {collection?.name ?? (loading ? m["collections.loading"]() : collectionId)}
            </h3>
            <p className="font-mono text-caption text-on-surface-variant truncate">
              {collectionId}
            </p>
          </div>
        </div>
        {locked && (
          <span
            title={m["board.locked"]()}
            aria-label={m["board.locked"]()}
            data-testid="lock-indicator"
          >
            <Lock className="h-3.5 w-3.5 text-on-surface-variant shrink-0" aria-hidden="true" />
          </span>
        )}
      </div>

      <div className="mt-3 flex items-center justify-between border-t border-outline/50 pt-2 text-caption text-on-surface-variant">
        <span>
          {slotCount} {slotLabel}
        </span>
        {collection && <span>v{collection.version}</span>}
      </div>
    </div>
  );
}

export function AssetRefCard({ kind, id, locked }: BoardCardProps) {
  const artifactId = (kind as { type: "asset_ref"; artifact_id: string }).artifact_id;

  return (
    <div
      className="flex h-full w-full flex-col justify-between rounded-xl border border-outline bg-surface p-4 text-on-surface shadow-xs transition-shadow select-none hover:shadow-md"
      data-testid="asset-ref-card"
    >
      <div className="flex items-start justify-between gap-2.5">
        <div className="flex items-center gap-2.5 min-w-0 flex-1">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-accent/10 text-accent">
            <FileImage className="h-4 w-4" />
          </div>
          <div className="min-w-0 flex-1">
            <span className="text-caption font-medium text-on-surface-variant">
              {m["board.asset"]()}
            </span>
            <p className="truncate font-mono text-body-sm font-medium text-on-surface">
              {artifactId}
            </p>
          </div>
        </div>
        {locked && (
          <span
            title={m["board.locked"]()}
            aria-label={m["board.locked"]()}
            data-testid="lock-indicator"
          >
            <Lock className="h-3.5 w-3.5 text-on-surface-variant shrink-0" aria-hidden="true" />
          </span>
        )}
      </div>
      <div className="mt-2 text-caption text-on-surface-variant truncate font-mono">id: {id}</div>
    </div>
  );
}

export function WorkflowRefCard({ kind, id, locked }: BoardCardProps) {
  const workflowId = (kind as { type: "workflow_ref"; workflow_id: string }).workflow_id;

  return (
    <div
      className="flex h-full w-full flex-col justify-between rounded-xl border border-outline bg-surface p-4 text-on-surface shadow-xs transition-shadow select-none hover:shadow-md"
      data-testid="workflow-ref-card"
    >
      <div className="flex items-start justify-between gap-2.5">
        <div className="flex items-center gap-2.5 min-w-0 flex-1">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-secondary/10 text-secondary">
            <Boxes className="h-4 w-4" />
          </div>
          <div className="min-w-0 flex-1">
            <span className="text-caption font-medium text-on-surface-variant">
              {m["board.workflow"]()}
            </span>
            <p className="truncate font-mono text-body-sm font-medium text-on-surface">
              {workflowId}
            </p>
          </div>
        </div>
        {locked && (
          <span
            title={m["board.locked"]()}
            aria-label={m["board.locked"]()}
            data-testid="lock-indicator"
          >
            <Lock className="h-3.5 w-3.5 text-on-surface-variant shrink-0" aria-hidden="true" />
          </span>
        )}
      </div>
      <div className="mt-2 text-caption text-on-surface-variant truncate font-mono">id: {id}</div>
    </div>
  );
}

export function RunRefCard({ kind, id, locked }: BoardCardProps) {
  const runId = (kind as { type: "run_ref"; run_id: string }).run_id;
  const navigate = useNavigate();

  return (
    <div
      onClick={() => void navigate({ to: "/library", search: { type: "run" } })}
      className="flex h-full w-full cursor-pointer flex-col justify-between rounded-xl border border-outline bg-surface p-4 text-on-surface shadow-xs transition-all select-none hover:border-primary/50 hover:bg-surface-container-high hover:shadow-md"
      data-testid="run-ref-card"
    >
      <div className="flex items-start justify-between gap-2.5">
        <div className="flex items-center gap-2.5 min-w-0 flex-1">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-500">
            <Play className="h-4 w-4" />
          </div>
          <div className="min-w-0 flex-1">
            <span className="text-caption font-medium text-on-surface-variant">
              {m["board.run"]()}
            </span>
            <p className="truncate font-mono text-body-sm font-medium text-on-surface">{runId}</p>
          </div>
        </div>
        {locked && (
          <span
            title={m["board.locked"]()}
            aria-label={m["board.locked"]()}
            data-testid="lock-indicator"
          >
            <Lock className="h-3.5 w-3.5 text-on-surface-variant shrink-0" aria-hidden="true" />
          </span>
        )}
      </div>
      <div className="mt-2 text-caption text-on-surface-variant truncate font-mono">id: {id}</div>
    </div>
  );
}

export function PlaceholderCard({ kind, id, locked }: BoardCardProps) {
  const typeName = kind?.type ?? "unknown";

  return (
    <div
      className="flex h-full w-full flex-col justify-between rounded-xl border border-dashed border-outline bg-surface/50 p-4 text-on-surface shadow-xs transition-shadow select-none"
      data-testid="placeholder-card"
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <span className="rounded bg-control-hover px-2 py-0.5 font-mono text-caption font-medium text-on-surface-variant">
            {typeName}
          </span>
          <p className="mt-2 text-body-sm text-on-surface-variant">
            {m["board.unregisteredItem"]()}
          </p>
        </div>
        {locked && (
          <span
            title={m["board.locked"]()}
            aria-label={m["board.locked"]()}
            data-testid="lock-indicator"
          >
            <Lock className="h-3.5 w-3.5 text-on-surface-variant shrink-0" aria-hidden="true" />
          </span>
        )}
      </div>
      <div className="mt-2 font-mono text-caption text-on-surface-variant truncate">id: {id}</div>
    </div>
  );
}

const REGISTRY: Record<string, React.ComponentType<BoardCardProps>> = {
  note: NoteCard,
  collection: CollectionCard,
  asset_ref: AssetRefCard,
  workflow_ref: WorkflowRefCard,
  run_ref: RunRefCard,
};

export function getCardRenderer(type: string): React.ComponentType<BoardCardProps> {
  return REGISTRY[type] ?? PlaceholderCard;
}
