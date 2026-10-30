/**
 * Design tokens — runtime constants for the UI.
 *
 * Source of truth: ui/src/styles/theme.css (@theme block). Values that exist
 * in both files MUST stay in sync; drift between the two is a bug.
 */

/** Shell layout dimensions in px, mirrored by the --spacing-* shell tokens
 *  in theme.css. */
export const LAYOUT = {
  sidebar: 264,
  panel: 364,
  panelGap: 16,
  topbar: 48,
} as const;

/** Icon stroke width used by the shell. */
export const ICON_STROKE = 1.5;
