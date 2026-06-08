import { ChevronDown } from "lucide-react";
import { type CSSProperties, type ReactNode } from "react";
import { cn } from "@/lib/utils";
import { Socket } from "./Socket";
import { SOCKET_COLORS, type SocketKind } from "@/design/tokens";

export type SocketSlot = {
  id: string;
  kind: SocketKind;
  /** Parameter name shown next to the socket (e.g. "model", "positive"). */
  label: string;
  /** Override the socket dot color (defaults to the type color). */
  dotColor?: string;
  /** Optional type badge rendered as a small square next to the handle (e.g. "F", "V"). */
  badge?: string;
};

export type ParamRow = {
  id: string;
  label: string;
  value: string;
  /** Optional secondary chip rendered next to the value (e.g. a "Kate" tag). */
  tag?: string;
};

type Props = {
  title: string;
  /** Hex color that drives the header dot, selected ring, and outer glow. */
  tone: string;
  inputs?: SocketSlot[];
  outputs?: SocketSlot[];
  parameters?: ParamRow[];
  children?: ReactNode;
  selected?: boolean;
};

/**
 * BaseNode — visual scaffold every node in the editor shares.
 *
 * Three structural layers, each with a distinct surface treatment:
 *
 *   1. .node-base       — outer frosted-glass card; lets the canvas dot
 *                         grid show through (blurred). One per node.
 *   2. .node-base__header — its own sub-container, faint white wash +
 *                         hairline divider. Holds the tone dot + title.
 *   3. .node-base__body — the shared container for sockets + inner card.
 *                         No background of its own.
 *   4. .node-base__inner — the deepest "well", holds children / parameters.
 *
 * `style` sets the per-instance `--node-tone` CSS custom property, which
 * `.node-base--selected` reads for the border, the 1px ring, and the
 * outer glow. The header dot is driven by inline style so the tone can
 * be set per node without a new CSS class.
 */
export function BaseNode({
  title,
  tone,
  inputs = [],
  outputs = [],
  parameters = [],
  children,
  selected,
}: Props) {
  const hasSockets = inputs.length > 0 || outputs.length > 0;
  const hasBody = children != null || parameters.length > 0;
  const hasContent = hasSockets || hasBody;

  return (
    <div
      className={cn("node-base", selected && "node-base--selected")}
      style={{ "--node-tone": tone } as CSSProperties}
    >
      {/* Header — own sub-container */}
      <div className="node-base__header">
        <span
          className="h-2 w-2 shrink-0 rounded-full"
          style={{
            backgroundColor: tone,
            boxShadow: `0 0 8px ${tone}`,
          }}
        />
        <span className="truncate text-[13px] font-medium leading-none text-zinc-200">
          {title}
        </span>
      </div>

      {/* Body — sockets + inner share this container */}
      {hasContent && (
        <div className="node-base__body">
          {hasSockets && (
            <div className="flex justify-between gap-2">
              <div className="flex flex-1 flex-col gap-1.5">
                {inputs.map((slot) => (
                  <div
                    key={slot.id}
                    className="relative flex h-5 w-full items-center gap-2"
                  >
                    <Socket
                      id={slot.id}
                      kind={slot.kind}
                      side="left"
                      dotColor={slot.dotColor}
                    />
                    {slot.badge && (
                      <span
                        className="flex h-4 w-4 shrink-0 items-center justify-center rounded text-[9px] font-bold text-white"
                        style={{
                          backgroundColor:
                            slot.dotColor || SOCKET_COLORS[slot.kind],
                        }}
                      >
                        {slot.badge}
                      </span>
                    )}
                    <span className="text-[11px] leading-none text-zinc-400">
                      {slot.label}
                    </span>
                  </div>
                ))}
              </div>
              <div className="flex flex-1 flex-col items-end gap-1.5">
                {outputs.map((slot) => (
                  <div
                    key={slot.id}
                    className="relative flex h-5 w-full items-center justify-end gap-2"
                  >
                    <span className="text-[11px] leading-none text-zinc-400">
                      {slot.label}
                    </span>
                    {slot.badge && (
                      <span
                        className="flex h-4 w-4 shrink-0 items-center justify-center rounded text-[9px] font-bold text-white"
                        style={{
                          backgroundColor:
                            slot.dotColor || SOCKET_COLORS[slot.kind],
                        }}
                      >
                        {slot.badge}
                      </span>
                    )}
                    <Socket
                      id={slot.id}
                      kind={slot.kind}
                      side="right"
                      dotColor={slot.dotColor}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}

          {hasBody && (
            <div className="node-base__inner">
              {children}
              {parameters.length > 0 && (
                <div className="space-y-1.5">
                  {parameters.map((p) => (
                    <div
                      key={p.id}
                      className="flex items-center justify-between gap-2"
                    >
                      {p.label && (
                        <span className="text-[11px] leading-none text-zinc-400">
                          {p.label}
                        </span>
                      )}
                      <div className="flex items-center gap-1.5">
                        <span className="flex items-center gap-1.5 rounded-md bg-zinc-800/70 px-2.5 py-1.5 text-[11px] font-medium leading-none text-zinc-100">
                          <span className="truncate">{p.value}</span>
                          <ChevronDown className="h-3 w-3 shrink-0 text-zinc-500" />
                        </span>
                        {p.tag && (
                          <span className="rounded-md bg-zinc-700/60 px-2 py-1.5 text-[10px] font-medium leading-none text-zinc-300">
                            {p.tag}
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}