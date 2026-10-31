import { useEffect, useRef, useState, type ReactNode } from "react";
import { CircleAlert, Plus, Trash2 } from "lucide-react";
import { diffLines } from "diff";
import { toast } from "sonner";
import * as m from "$paraglide/messages";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { opInvoke, opPreview, readAssetText } from "@/ipc";
import { CollectionDocumentSchema, type CollectionSlot, type OperationInfo } from "@/ipc/schemas";
import { getPreviewUrl } from "@/lib/assetUrl";

const PREVIEW_DEBOUNCE_MS = 275;

type JsonSchema = {
  type?: string | string[];
  properties?: Record<string, JsonSchema>;
  items?: JsonSchema;
  enum?: unknown[];
  default?: unknown;
  format?: string;
};

type PairRow = {
  key: string;
  values: Record<string, unknown>;
};

type PreviewResult = { text: string; counts: number[] };

let pairSequence = 0;

function nextPairKey(): string {
  pairSequence += 1;
  return `pair-${pairSequence}`;
}

/** A preview key is stable per dialog and safe as a directory name. */
function createPreviewKey(): string {
  return `pk_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

function schemaOf(operation: OperationInfo | undefined): JsonSchema {
  return (operation?.spec?.input_schema ?? {}) as JsonSchema;
}

/** Defaults come from the pair item schema so the form follows the operation. */
function defaultPairValues(items: JsonSchema | undefined): Record<string, unknown> {
  const values: Record<string, unknown> = {};
  for (const [name, field] of Object.entries(items?.properties ?? {})) {
    const type = Array.isArray(field.type) ? field.type[0] : field.type;
    if (type === "boolean") {
      values[name] = typeof field.default === "boolean" ? field.default : false;
    } else if (Array.isArray(field.enum)) {
      values[name] = field.default ?? field.enum[0] ?? "";
    } else {
      values[name] = typeof field.default === "string" ? field.default : "";
    }
  }
  if (typeof values.mode !== "string") values.mode = "literal";
  if (typeof values.case_sensitive !== "boolean") values.case_sensitive = true;
  return values;
}

function createPair(items: JsonSchema | undefined): PairRow {
  return { key: nextPairKey(), values: defaultPairValues(items) };
}

function pairValue(pair: PairRow, name: string): unknown {
  return pair.values[name];
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function fieldLabel(name: string): string {
  switch (name) {
    case "find":
      return m["collections.textTool.fields.find"]();
    case "replace":
      return m["collections.textTool.fields.replace"]();
    case "mode":
      return m["collections.textTool.fields.mode"]();
    case "case_sensitive":
      return m["collections.textTool.fields.caseSensitive"]();
    default:
      return name;
  }
}

function modeLabel(mode: string): string {
  switch (mode) {
    case "word":
      return m["collections.textTool.modes.word"]();
    case "regex":
      return m["collections.textTool.modes.regex"]();
    default:
      return m["collections.textTool.modes.literal"]();
  }
}

function pairsForInput(pairs: PairRow[]): Array<Record<string, unknown>> {
  return pairs
    .filter((pair) => String(pairValue(pair, "find") ?? "").length > 0)
    .map((pair) => ({
      find: String(pairValue(pair, "find") ?? ""),
      replace: String(pairValue(pair, "replace") ?? ""),
      mode: String(pairValue(pair, "mode") ?? "literal"),
      case_sensitive: pairValue(pair, "case_sensitive") !== false,
    }));
}

/**
 * Resolve a preview output: the mock returns inline `text`, the host returns a
 * `doc` URI that has to be read through the asset protocol.
 */
async function readPreviewOutput(output: unknown): Promise<PreviewResult> {
  if (!output || typeof output !== "object") {
    throw new Error("preview returned no output");
  }
  const record = output as { text?: unknown; doc?: unknown; counts?: unknown };
  const counts = Array.isArray(record.counts)
    ? record.counts.map((count) => (typeof count === "number" ? count : 0))
    : [];
  if (typeof record.text === "string") {
    return { text: record.text, counts };
  }
  if (typeof record.doc === "string") {
    const res = await fetch(getPreviewUrl(record.doc));
    if (!res.ok) throw new Error(`HTTP ${res.status} reading the preview`);
    return { text: await res.text(), counts };
  }
  throw new Error("preview returned neither text nor a document");
}

function DiffView({ original, changed }: { original: string; changed: string }) {
  const parts = diffLines(original, changed);
  return (
    <pre
      data-testid="text-tool-diff"
      className="max-h-80 overflow-auto rounded-lg bg-tile p-3 font-mono text-caption whitespace-pre-wrap text-fg"
    >
      {parts.map((part, partIndex) => {
        const lines = part.value.split("\n");
        if (lines[lines.length - 1] === "") lines.pop();
        return lines.map((line, lineIndex) => {
          const tone = part.added
            ? "bg-ok/10 text-fg"
            : part.removed
              ? "bg-err/10 text-fg"
              : "text-fg2";
          const marker = part.added ? "+ " : part.removed ? "- " : "  ";
          return (
            <div
              key={`${partIndex}-${lineIndex}`}
              className={tone}
              data-diff={part.added ? "added" : part.removed ? "removed" : "unchanged"}
            >
              <span aria-hidden="true">{marker}</span>
              {line}
            </div>
          );
        });
      })}
    </pre>
  );
}

function Diagnostic({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-start gap-2 rounded-lg border-l-2 border-err bg-err/10 p-3 text-body-sm text-fg">
      <CircleAlert className="mt-0.5 size-4 shrink-0 text-err" aria-hidden="true" />
      <span>{children}</span>
    </div>
  );
}

export function TextToolDialog({
  open,
  projectId,
  collectionId,
  slot,
  operation,
  onOpenChange,
  onApplied,
}: {
  open: boolean;
  projectId: string;
  collectionId: string;
  slot: CollectionSlot;
  operation: OperationInfo | undefined;
  onOpenChange: (open: boolean) => void;
  onApplied?: () => void | Promise<void>;
}) {
  const inputSchema = schemaOf(operation);
  const pairsSchema = inputSchema.properties?.pairs;
  const pairItems = pairsSchema?.items;

  const [pairs, setPairs] = useState<PairRow[]>(() => [createPair(pairItems)]);
  const [original, setOriginal] = useState<string | null>(null);
  const [originalError, setOriginalError] = useState<string | null>(null);
  const [preview, setPreview] = useState<PreviewResult | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [applying, setApplying] = useState(false);
  const [applyError, setApplyError] = useState<string | null>(null);

  const previewKeyRef = useRef(createPreviewKey());
  const previewSequence = useRef(0);
  const assetId = slot.asset_id;

  // A new open starts from a clean form, a fresh preview key and no late results.
  useEffect(() => {
    if (!open) {
      previewSequence.current += 1;
      return;
    }
    previewSequence.current += 1;
    previewKeyRef.current = createPreviewKey();
    setPairs([createPair(pairItems)]);
    setOriginal(null);
    setOriginalError(null);
    setPreview(null);
    setPreviewError(null);
    setPreviewing(false);
    setApplying(false);
    setApplyError(null);
  }, [open, projectId, assetId, pairItems]);

  useEffect(() => {
    if (!open) return;
    let active = true;
    readAssetText(projectId, assetId).then(
      (text) => {
        if (active) setOriginal(text);
      },
      (error: unknown) => {
        if (active) setOriginalError(errorMessage(error));
      },
    );
    return () => {
      active = false;
    };
  }, [open, projectId, assetId]);

  const validPairs = pairsForInput(pairs);
  const hasPairs = validPairs.length > 0;

  useEffect(() => {
    const sequence = ++previewSequence.current;
    if (!open || original === null || !hasPairs) {
      setPreview(null);
      setPreviewError(null);
      setPreviewing(false);
      return;
    }
    setPreviewing(true);
    setPreviewError(null);
    const timer = setTimeout(() => {
      void (async () => {
        try {
          const output = await opPreview(
            projectId,
            "text.replace",
            { doc: assetId, pairs: validPairs },
            previewKeyRef.current,
          );
          if (sequence !== previewSequence.current) return;
          const result = await readPreviewOutput(output);
          if (sequence !== previewSequence.current) return;
          setPreview(result);
          setPreviewing(false);
        } catch (error) {
          if (sequence !== previewSequence.current) return;
          setPreview(null);
          setPreviewError(errorMessage(error));
          setPreviewing(false);
        }
      })();
    }, PREVIEW_DEBOUNCE_MS);

    return () => {
      clearTimeout(timer);
    };
    // `validPairs` is derived from `pairs`; the signature keeps the effect
    // from re-running for edits that leave the payload unchanged.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, original, pairs, projectId, assetId, hasPairs]);

  function updatePair(key: string, name: string, value: unknown) {
    setPairs((current) =>
      current.map((pair) =>
        pair.key === key ? { ...pair, values: { ...pair.values, [name]: value } } : pair,
      ),
    );
  }

  function addPair() {
    setPairs((current) => [...current, createPair(pairItems)]);
  }

  function removePair(key: string) {
    setPairs((current) => current.filter((pair) => pair.key !== key));
  }

  function handleOpenChange(next: boolean) {
    if (!next) previewSequence.current += 1;
    onOpenChange(next);
  }

  const canApply =
    hasPairs &&
    preview !== null &&
    !previewing &&
    !applying &&
    previewError === null &&
    originalError === null;

  async function handleApply() {
    if (!canApply) return;
    setApplying(true);
    setApplyError(null);
    try {
      const current = await opInvoke(projectId, "collection.get", {
        collection_id: collectionId,
      });
      const document = CollectionDocumentSchema.parse(current.output);
      const currentSlot = document.slots.find((candidate) => candidate.id === slot.id);
      if (!currentSlot || currentSlot.asset_id !== assetId) {
        setApplyError(m["collections.textTool.slotChanged"]());
        return;
      }

      const invoked = await opInvoke(projectId, "text.replace", {
        doc: assetId,
        pairs: validPairs,
      });
      const output = (invoked.output ?? null) as { doc?: unknown } | null;
      if (!output || typeof output.doc !== "string") {
        setApplyError(m["collections.textTool.inlineOutput"]());
        return;
      }

      await opInvoke(projectId, "collection.set_slot", {
        collection_id: collectionId,
        slot_id: slot.id,
        asset: output.doc,
      });

      toast.success(m["collections.textTool.applied"]());
      try {
        await onApplied?.();
      } catch {
        // The collection.changed event still refreshes an open view.
      }
      handleOpenChange(false);
    } catch (error) {
      setApplyError(`${m["collections.textTool.applyError"]()}: ${errorMessage(error)}`);
    } finally {
      setApplying(false);
    }
  }

  const unchanged = original !== null && preview !== null && preview.text === original;
  const previewText = preview?.text ?? original ?? "";

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-4xl">
        <DialogHeader>
          <DialogTitle>{m["collections.textTool.title"]()}</DialogTitle>
          <DialogDescription>
            {m["collections.textTool.description"]()} ({m["collections.slotPrefix"]()} {slot.id})
          </DialogDescription>
        </DialogHeader>

        {originalError ? (
          <Diagnostic>
            {m["collections.textTool.originalError"]()}: {originalError}
          </Diagnostic>
        ) : original === null ? (
          <div className="text-body-sm text-fg2">{m["collections.textTool.loadingOriginal"]()}</div>
        ) : !pairsSchema || !pairItems ? (
          <div className="text-body-sm text-fg2">{m["collections.textTool.unavailable"]()}</div>
        ) : (
          <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <section
              className="flex flex-col gap-3"
              aria-label={m["collections.textTool.fields.pairs"]()}
            >
              <div className="flex items-center justify-between">
                <h3 className="text-body-sm font-medium text-fg">
                  {m["collections.textTool.fields.pairs"]()}
                </h3>
                <Button type="button" variant="outline" size="sm" onClick={addPair}>
                  <Plus aria-hidden="true" />
                  {m["collections.textTool.addPair"]()}
                </Button>
              </div>

              {pairs.length === 0 && (
                <p className="text-caption text-fg2">{m["collections.textTool.noPairs"]()}</p>
              )}

              {pairs.map((pair, index) => {
                return (
                  <div
                    key={pair.key}
                    className="rounded-lg border border-hair bg-ground p-3"
                    data-testid="text-tool-pair"
                  >
                    <div className="mb-2 flex items-center justify-between">
                      <span className="text-caption font-medium text-fg2">
                        {m["collections.textTool.pairLabel"]({ index: index + 1 })}
                      </span>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon-xs"
                        aria-label={m["collections.textTool.removePair"]()}
                        onClick={() => removePair(pair.key)}
                      >
                        <Trash2 aria-hidden="true" />
                      </Button>
                    </div>

                    <div className="flex flex-col gap-2">
                      {Object.entries(pairItems.properties ?? {}).map(([name, field]) => {
                        const id = `text-tool-${pair.key}-${name.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
                        const type = Array.isArray(field.type) ? field.type[0] : field.type;
                        if (Array.isArray(field.enum)) {
                          return (
                            <div key={name} className="flex flex-col gap-2">
                              <label htmlFor={id} className="text-caption text-fg2">
                                {fieldLabel(name)}
                              </label>
                              <select
                                id={id}
                                value={String(pairValue(pair, name) ?? "")}
                                onChange={(event) => updatePair(pair.key, name, event.target.value)}
                                className="h-9 w-full rounded-md border border-hair bg-surface px-3 text-sm text-fg outline-none focus-visible:border-accent focus-visible:ring-[3px] focus-visible:ring-accent/25"
                              >
                                {field.enum.map((option) => (
                                  <option key={String(option)} value={String(option)}>
                                    {name === "mode" ? modeLabel(String(option)) : String(option)}
                                  </option>
                                ))}
                              </select>
                            </div>
                          );
                        }
                        if (type === "boolean") {
                          return (
                            <label
                              key={name}
                              htmlFor={id}
                              className="mt-1 flex items-center gap-2 text-body-sm text-fg"
                            >
                              <input
                                id={id}
                                type="checkbox"
                                checked={pairValue(pair, name) !== false}
                                onChange={(event) =>
                                  updatePair(pair.key, name, event.target.checked)
                                }
                                className="size-4 accent-accent"
                              />
                              {fieldLabel(name)}
                            </label>
                          );
                        }
                        if (type === "string" || type === undefined) {
                          return (
                            <div key={name} className="flex flex-col gap-2">
                              <label htmlFor={id} className="text-caption text-fg2">
                                {fieldLabel(name)}
                              </label>
                              <Input
                                id={id}
                                value={String(pairValue(pair, name) ?? "")}
                                onChange={(event) => updatePair(pair.key, name, event.target.value)}
                              />
                            </div>
                          );
                        }
                        return null;
                      })}
                    </div>
                  </div>
                );
              })}
            </section>

            <section
              className="flex min-w-0 flex-col gap-2"
              aria-label={m["collections.textTool.preview"]()}
            >
              <div className="flex items-center justify-between">
                <h3 className="text-body-sm font-medium text-fg">
                  {m["collections.textTool.preview"]()}
                </h3>
                {previewing && (
                  <span className="text-caption text-fg2">
                    {m["collections.textTool.previewing"]()}
                  </span>
                )}
              </div>

              {!hasPairs ? (
                <>
                  <p className="text-caption text-fg2">{m["collections.textTool.noPairs"]()}</p>
                  <pre className="max-h-80 overflow-auto rounded-lg bg-tile p-3 font-mono text-caption whitespace-pre-wrap text-fg2">
                    {original}
                  </pre>
                </>
              ) : previewError ? (
                <Diagnostic>
                  {m["collections.textTool.previewError"]()}: {previewError}
                </Diagnostic>
              ) : preview ? (
                <>
                  {unchanged && (
                    <p className="text-caption text-fg2">{m["collections.textTool.noChanges"]()}</p>
                  )}
                  <DiffView original={original} changed={previewText} />
                  <ul className="flex flex-col gap-1 text-caption text-fg2">
                    {preview.counts.map((count, index) => (
                      <li key={index}>
                        {m["collections.textTool.pairCount"]({
                          index: index + 1,
                          count,
                        })}
                      </li>
                    ))}
                  </ul>
                </>
              ) : (
                <p className="text-caption text-fg2">{m["collections.textTool.emptyPreview"]()}</p>
              )}
            </section>
          </div>
        )}

        {applyError && <Diagnostic>{applyError}</Diagnostic>}

        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={() => handleOpenChange(false)}
            disabled={applying}
          >
            {m["common.cancel"]()}
          </Button>
          <Button type="button" onClick={() => void handleApply()} disabled={!canApply}>
            {applying ? m["collections.textTool.applying"]() : m["collections.textTool.apply"]()}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
