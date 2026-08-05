import { Image as ImageIcon, Loader2, TriangleAlert } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ArtifactPreview } from "@/store/artifacts";

/**
 * NodePreview — inline thumbnail for image-producing nodes (F5-4).
 *
 * Renders the pending/ready/error/stale states of a run artifact. Used by
 * GenericNode and the hand-crafted ImageOutputNode.
 */
export function NodePreview({ preview }: { preview?: ArtifactPreview }) {
  if (!preview) {
    return (
      <div className="flex aspect-square w-full flex-col items-center justify-center gap-1.5 text-on-surface-variant">
        <ImageIcon className="h-4 w-4" />
        <span className="text-caption">No preview yet</span>
      </div>
    );
  }

  if (preview.status === "ready" && preview.url) {
    return (
      <img
        className="block aspect-square w-full object-cover"
        src={preview.url}
        alt={`Artifact ${preview.artifactId}`}
      />
    );
  }

  const message =
    preview.status === "pending"
      ? "Generating…"
      : preview.status === "error"
        ? "Preview failed"
        : "Run ended";

  return (
    <div
      className={cn(
        "flex aspect-square w-full flex-col items-center justify-center gap-1.5 text-on-surface-variant",
        preview.status === "error" && "text-status-error",
      )}
    >
      {preview.status === "pending" ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : preview.status === "error" ? (
        <TriangleAlert className="h-4 w-4" />
      ) : (
        <ImageIcon className="h-4 w-4" />
      )}
      <span className="text-caption">{message}</span>
    </div>
  );
}
