/**
 * Design tokens — runtime constants for the UI.
 *
 * Source of truth: ui/src/styles/theme.css (@theme block). Values that exist
 * in both files MUST stay in sync; drift between the two is a bug.
 */

/**
 * Socket type → CSS color. The custom properties are plain theme-block
 * variables in theme.css, so a socket's color follows the active theme.
 */
export const SOCKET_COLORS = {
  model: "var(--color-socket-model)", // model and checkpoint route
  conditioning: "var(--color-socket-conditioning)", // positive/negative conditioning route
  latent: "var(--color-socket-latent)", // latent tensor route
  image: "var(--color-socket-image)", // image artifact route
} as const;

export type SocketKind = keyof typeof SOCKET_COLORS;

/** Font stacks — match --font-sans / --font-mono in theme.css. */
export const FONT_STACKS = {
  sans: '"Geist", -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", system-ui, sans-serif',
  mono: '"Geist Mono", "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace',
} as const;

/** Layout dimensions in px. Mirrored as --spacing-* in theme.css. */
export const LAYOUT = {
  sidebar: 264,
  panel: 364,
  panelGap: 16,
  topbar: 48,
  sidebarHeader: 64,
  canvasGrid: 20,
  nodePadding: 12,
  gutterMd: 16,
  stackTight: 4,
  stackBase: 8,
  controlHeight: 32,
} as const;

/** Border radius. Mirrored as --radius-* in theme.css. */
export const RADIUS = {
  xs: "6px",
  sm: "8px",
  md: "10px",
  lg: "12px",
  xl: "16px",
  full: "9999px",
} as const;

/** Standardized icon sizes (lucide-react uses pixel size attribute). */
export const ICON_SIZE = {
  sm: 14,
  md: 16,
  lg: 20,
  xl: 24,
} as const;

/** Icon stroke width used by the shell. */
export const ICON_STROKE = 1.5;
