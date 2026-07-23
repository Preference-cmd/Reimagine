/**
 * Design tokens — runtime constants for the UI.
 *
 * Source of truth: Reimagine theme primitives in ui/src/styles/theme.css.
 * Mirrored to CSS in ui/src/styles/theme.css (@theme block).
 *
 * Rule: any value added here MUST also be added to theme.css in the
 * matching @theme variable, and vice versa. Drift between the two
 * is a bug.
 */

/** Socket type → color hex. Mirrored as --color-socket-* in theme.css. */
export const SOCKET_COLORS = {
  model:        '#f5a623',  // model and checkpoint route
  conditioning: '#f5a623',  // positive/negative conditioning route
  latent:       '#7928ca',  // latent tensor route
  image:        '#50e3c2',  // image artifact route
} as const;

export type SocketKind = keyof typeof SOCKET_COLORS;

/** Font stacks — must match --font-sans / --font-mono in theme.css. */
export const FONT_STACKS = {
  sans: '"Inter", system-ui, sans-serif',
  mono: '"JetBrains Mono", ui-monospace, monospace',
} as const;

/** Layout dimensions in px. Mirrored as --spacing-* in theme.css. */
export const LAYOUT = {
  canvasGrid:    20,
  nodePadding:   12,
  sidebarWidth:  280,
  gutterMd:      16,
  stackTight:     4,
  stackBase:      8,
  controlHeight: 32,
} as const;

/** Border radius. Mirrored as --radius-* in theme.css. */
export const RADIUS = {
  sm:      '0.125rem',
  DEFAULT: '0.25rem',
  md:      '0.375rem',
  lg:      '0.5rem',
  xl:      '0.75rem',
  full:    '9999px',
} as const;

/** Standardized icon sizes (lucide-react uses pixel size attribute). */
export const ICON_SIZE = {
  sm: 14,
  md: 16,
  lg: 20,
  xl: 24,
} as const;
