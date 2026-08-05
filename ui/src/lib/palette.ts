/**
 * Command palette helpers (F3-3): fuzzy matching over command/node lists.
 * Pure and store-free so the filtering logic stays unit-testable.
 */

export type PaletteEntry = {
  id: string;
  label: string;
  hint?: string;
  keywords?: string[];
  shortcut?: string;
};

/**
 * Subsequence match score for `query` inside `text` (case-insensitive).
 * Returns 0 when the query is not a subsequence; higher scores rank better:
 * consecutive runs, word-boundary starts and prefix matches win.
 */
export function fuzzyScore(query: string, text: string): number {
  const needle = query.trim().toLowerCase();
  if (!needle) return 0;
  const haystack = text.toLowerCase();
  if (haystack.startsWith(needle)) return 1000 + needle.length;
  if (haystack.includes(needle)) return 500 + needle.length;

  let score = 0;
  let needleIndex = 0;
  let streak = 0;
  let prevMatch = -2;
  for (let i = 0; i < haystack.length && needleIndex < needle.length; i += 1) {
    if (haystack[i] !== needle[needleIndex]) {
      streak = 0;
      continue;
    }
    needleIndex += 1;
    streak += 1;
    score += streak > 1 ? 10 : 1;
    if (i === prevMatch + 1) score += 5;
    const boundary = i === 0 || /[\s-_/:.]/.test(haystack[i - 1]);
    if (boundary) score += 3;
    prevMatch = i;
  }
  return needleIndex === needle.length ? score : 0;
}

/** Score an entry against a query across label, hint and keywords. */
export function scoreEntry(query: string, entry: PaletteEntry): number {
  let best = fuzzyScore(query, entry.label);
  if (entry.hint) best = Math.max(best, fuzzyScore(query, entry.hint) - 1);
  for (const keyword of entry.keywords ?? []) {
    best = Math.max(best, fuzzyScore(query, keyword) - 1);
  }
  return best;
}

/** Filter + rank entries by fuzzy relevance; empty query keeps order. */
export function filterEntries<T extends PaletteEntry>(
  query: string,
  entries: T[],
): T[] {
  if (!query.trim()) return entries;
  const scored = entries
    .map((entry) => ({ entry, score: scoreEntry(query, entry) }))
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score);
  return scored.map(({ entry }) => entry);
}
