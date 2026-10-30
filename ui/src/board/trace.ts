import type { CanvasTrace } from "@/store/uiStore";

/**
 * Ring for items an activity record touched. Agent work is gold; every other
 * actor (human, importer, system) keeps the workspace accent.
 */
const TRACED_HUMAN = "rounded-lg shadow-[0_0_0_2px_var(--color-accent)]";
const TRACED_AGENT = "rounded-lg shadow-[0_0_0_2px_var(--color-agent)]";

/** Everything else fades while a trace is active. */
const TRACED_DIM = "opacity-40";

/**
 * Extra class for a board node wrapper under the panel's hover trace.
 * Returns an empty string when no trace is active, so the canvas looks
 * exactly as it does without the activity panel.
 */
export function boardNodeTraceClass(trace: CanvasTrace | null, itemId: string): string {
  if (!trace) return "";
  if (!trace.targets.includes(itemId)) return TRACED_DIM;
  return trace.actorKind === "Agent" ? TRACED_AGENT : TRACED_HUMAN;
}
