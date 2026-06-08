import { Handle, Position, type HandleProps } from "@xyflow/react";
import { cn } from "@/lib/utils";
import { SOCKET_COLORS, type SocketKind } from "@/design/tokens";

type Props = Omit<HandleProps, "type" | "position"> & {
  kind: SocketKind;
  side: "left" | "right";
  connected?: boolean;
  /** Override the socket's color (defaults to the type color). */
  dotColor?: string;
};

/**
 * Socket — a small square handle sitting flush on the card edge.
 *
 * React Flow draws edges from this handle. It carries no text;
 * the type badge ("F", "V", etc.) is rendered as a separate element
 * inside the socket row, between the handle and the label.
 */
export function Socket({
  kind,
  side,
  connected = false,
  dotColor,
  style,
  ...rest
}: Props) {
  const color = dotColor || SOCKET_COLORS[kind];

  return (
    <Handle
      type={side === "left" ? "target" : "source"}
      position={side === "left" ? Position.Left : Position.Right}
      className={cn(
        "!h-2 !w-[5px] !rounded-sm !border-0",
        "transition-[box-shadow,transform] duration-200",
        "hover:scale-125",
      )}
      style={{
        top: "50%",
        transform: "translateY(-50%)",
        ...(side === "left"
          ? { left: "-19px", right: "auto" }
          : { right: "-19px", left: "auto" }),
        backgroundColor: color,
        boxShadow: connected
          ? `0 0 0 2px ${color}33, 0 0 10px ${color}88`
          : "none",
        ...style,
      }}
      isConnectable
      {...rest}
    />
  );
}