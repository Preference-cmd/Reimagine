import { type FC } from "react";
import {
  getBezierPath,
  type EdgeProps,
  type Edge,
} from "@xyflow/react";

export type FlowEdgeData = {
  /** Socket kind on the source side (drives the gradient start color). */
  sourceKind: string;
  /** Socket kind on the target side (drives the gradient end color). */
  targetKind: string;
  /** Optional pill label rendered at the midpoint (e.g. a name). */
  label?: string;
  /** Tone for the label pill; falls back to the source gradient stop. */
  tone?: string;
  [key: string]: unknown;
};

export type FlowEdge = Edge<FlowEdgeData, "flow">;

/**
 * Custom edge — gradient bezier with a flowing dash overlay and an
 * optional midpoint label pill, mirroring the ref.html connection-line
 * treatment.
 */
export const FlowEdgeComponent: FC<EdgeProps<FlowEdge>> = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  selected,
}) => {
  const [path, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    curvature: 0.35,
  });

  const startColor = data?.sourceKind
    ? `var(--color-socket-${data.sourceKind}, #a855f7)`
    : "#a855f7";
  const endColor = data?.targetKind
    ? `var(--color-socket-${data.targetKind}, #22c55e)`
    : "#22c55e";
  const labelColor = data?.tone || startColor;
  const glowId = `glow-${id}`;

  return (
    <>
      <defs>
        <linearGradient
          id={`grad-${id}`}
          gradientUnits="userSpaceOnUse"
          x1={sourceX}
          y1={sourceY}
          x2={targetX}
          y2={targetY}
        >
          <stop offset="0%" stopColor={startColor} stopOpacity="0.9" />
          <stop offset="100%" stopColor={endColor} stopOpacity="0.9" />
        </linearGradient>
        <filter id={glowId} x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur stdDeviation="2.5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Main line — thicker, with a soft glow */}
      <path
        d={path}
        fill="none"
        stroke={`url(#grad-${id})`}
        strokeWidth={selected ? 2.5 : 2}
        style={{ filter: `url(#${glowId})` }}
        className="pointer-events-none"
      />

      {/* Flowing dash overlay — brighter, more visible */}
      <path
        d={path}
        fill="none"
        stroke="#ffffff"
        strokeWidth={1.5}
        strokeDasharray="6 10"
        strokeOpacity={0.55}
        className="pointer-events-none"
        style={{ animation: "flow 1.2s linear infinite" }}
      />

      {/* Midpoint label pill — frosted-glass style, matching the node aesthetic */}
      {data?.label && (
        <g
          transform={`translate(${labelX}, ${labelY})`}
          className="pointer-events-none"
        >
          <rect
            x={-30}
            y={-10}
            width={60}
            height={20}
            rx={10}
            fill="rgba(20, 20, 20, 0.75)"
            stroke={labelColor}
            strokeOpacity={0.5}
            strokeWidth={1}
          />
          <text
            x={0}
            y={4.5}
            textAnchor="middle"
            fill={labelColor}
            fontSize={10}
            fontWeight={500}
            fontFamily="Inter, system-ui, sans-serif"
          >
            {data.label}
          </text>
        </g>
      )}

      <style>{`
        @keyframes flow {
          to { stroke-dashoffset: -32; }
        }
      `}</style>
    </>
  );
};

export const edgeTypes = {
  flow: FlowEdgeComponent,
};