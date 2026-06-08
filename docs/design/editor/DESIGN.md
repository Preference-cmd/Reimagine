---
name: Obsidian Node System
colors:
  surface: '#131313'
  surface-dim: '#131313'
  surface-bright: '#393939'
  surface-container-lowest: '#0e0e0e'
  surface-container-low: '#1c1b1b'
  surface-container: '#201f1f'
  surface-container-high: '#2a2a2a'
  surface-container-highest: '#353534'
  on-surface: '#e5e2e1'
  on-surface-variant: '#cbc3d7'
  inverse-surface: '#e5e2e1'
  inverse-on-surface: '#313030'
  outline: '#958ea0'
  outline-variant: '#494454'
  surface-tint: '#d0bcff'
  primary: '#d0bcff'
  on-primary: '#3c0091'
  primary-container: '#a078ff'
  on-primary-container: '#340080'
  inverse-primary: '#6d3bd7'
  secondary: '#4cd7f6'
  on-secondary: '#003640'
  secondary-container: '#03b5d3'
  on-secondary-container: '#00424e'
  tertiary: '#ffb0cd'
  on-tertiary: '#640039'
  tertiary-container: '#f751a1'
  on-tertiary-container: '#570032'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#e9ddff'
  primary-fixed-dim: '#d0bcff'
  on-primary-fixed: '#23005c'
  on-primary-fixed-variant: '#5516be'
  secondary-fixed: '#acedff'
  secondary-fixed-dim: '#4cd7f6'
  on-secondary-fixed: '#001f26'
  on-secondary-fixed-variant: '#004e5c'
  tertiary-fixed: '#ffd9e4'
  tertiary-fixed-dim: '#ffb0cd'
  on-tertiary-fixed: '#3e0022'
  on-tertiary-fixed-variant: '#8c0053'
  background: '#131313'
  on-background: '#e5e2e1'
  surface-variant: '#353534'
typography:
  headline-lg:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  code-md:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 18px
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '700'
    lineHeight: 12px
    letterSpacing: 0.05em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  canvas-grid: 20px
  node-padding: 12px
  sidebar-width: 280px
  gutter-md: 16px
  stack-tight: 4px
  stack-base: 8px
---

## Brand & Style

This design system is engineered for power users, developers, and AI artists who require a high-density, immersive workspace for complex logical orchestration. The aesthetic is defined as **Futuristic Technical Minimalism**, blending the raw utility of a terminal with the sophisticated depth of modern glassmorphism.

The UI is intentionally dark to reduce eye strain during long-duration focus sessions. It leverages a "Pro-Tool" narrative—where the interface recedes into the background to prioritize the user's workflow. Visual hierarchy is established through luminous accents and varying obsidian-toned surfaces rather than heavy decorative elements. The emotional response is one of precision, control, and infinite scalability.

## Colors

The palette is rooted in an ultra-dark foundation to maximize the "glow" effect of functional elements.

- **Foundation:** Deep Charcoal (#0A0A0A) serves as the infinite canvas background, while Obsidian (#121212) defines the primary containers and sidebars.
- **Accents:** Vibrant Purple (#8B5CF6) is used for primary actions and logical flow grouping. Cyan (#06B6D4) is reserved for data input/output ports and active connection lines, providing a high-contrast "circuit" feel.
- **Functional Tints:** Subtle pink and amber accents are used sparingly for specialized node categories (e.g., VAE or Post-processing) to allow for instant visual parsing of complex graphs.
- **Translucency:** Sidebars and floating panels utilize 70-80% opacity with a 20px background blur to maintain spatial awareness of the node graph behind them.

## Typography

The system employs a dual-font strategy to balance interface clarity with technical precision.

- **Inter** is the workhorse for the global UI, providing excellent legibility for property inspectors, navigation, and modal dialogues.
- **JetBrains Mono** is utilized for all node-specific data, including port labels, code blocks, parameter values, and terminal outputs. This reinforces the "IDE-like" nature of the application.
- **Information Density:** Large type is avoided within the canvas. Most node labels use `label-caps` for structural headers and `code-md` for interactive values to maximize the information visible on-screen without zooming.

## Layout & Spacing

The layout is centered around an **Infinite Canvas Model** with fixed-width supporting panels.

- **Canvas:** Features a dot-grid pattern spaced at 20px intervals. All nodes snap to this grid to maintain visual order.
- **Sidebars:** Fixed at 280px to accommodate complex property inspectors without obscuring the workflow.
- **Grids:** Internal component layouts use a 4px baseline. Buttons, inputs, and list items are standardized at 32px height to ensure high density in sidebars while remaining touch-targets for stylus users.
- **Responsive Behavior:** On smaller screens, sidebars collapse into icons, and the property inspector shifts to a floating bottom sheet.

## Elevation & Depth

Visual hierarchy is managed through **Luminance Stacking** rather than traditional drop shadows.

- **Level 0 (Canvas):** Pure black or #0A0A0A.
- **Level 1 (Nodes/Sidebars):** #121212. These elements feature a 1px solid border of #1A1A1A or a subtle glow if active.
- **Level 2 (Modals/Popovers):** #1A1A1A. These use a medium-diffusion ambient shadow (Black, 40% opacity, 12px blur) to appear physically separated from the canvas.
- **Interaction Depth:** When a node is dragged, it scales slightly (1.02x) and gains a Cyan (#06B6D4) outer glow to indicate its active state in the z-index stack.

## Shapes

The design system uses a **Soft Geometry** approach. Sharp corners are avoided to keep the technical UI from feeling aggressive, but large radii are rejected to maintain a professional tool aesthetic.

- **Nodes:** Use a 0.5rem (rounded-lg) corner radius for the main container and 0.25rem (soft) for internal header areas.
- **Input Fields:** Use 0.25rem radius to maximize internal space for text.
- **Connection Ports:** Perfectly circular (pill-shaped) to distinguish them from structural UI elements.

## Components

### Node Cards
The core component. Nodes consist of a header area with a category color-strip (top), a body area for inputs/parameters, and side-mounted ports. Ports should glow with the color of the data type they represent.

### Buttons & Inputs
- **Primary Action:** Solid purple background with white text.
- **Ghost Action:** Border-only with subtle hover fills.
- **Technical Inputs:** Darker than the surface color (#050505) with 1px borders. Sliders use a cyan track to indicate progress or value levels.

### Connection Lines
Bezier curves with a 2px stroke width. The stroke uses a gradient from the output port color to the input port color. When a flow is "running," an animated dash-array or "pulse" effect travels along the line.

### Glass Panels
Sidebars use a `backdrop-filter: blur(20px)` with a very thin white-translucent top border to simulate a glass edge catching the light.

### Chips & Badges
Small, monochromatic labels using JetBrains Mono. Used for tagging node versions (e.g., "v1.0.2") or status indicators (e.g., "Cached").