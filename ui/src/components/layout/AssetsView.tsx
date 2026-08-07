import { Folder, Image, Upload } from "lucide-react";

/**
 * AssetsView — full-page artifact browser.
 * Placeholder for now; will show generated images and imported assets.
 */
export function AssetsView() {
  return (
    <div className="flex h-full flex-col bg-background p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-display-sm font-semibold text-on-surface">
          Assets
        </h1>
        <p className="mt-1 text-body-sm text-on-surface-variant">
          Project files & generated images
        </p>
      </div>

      {/* Empty state */}
      <div className="flex min-h-0 flex-1 flex-col items-center justify-center text-on-surface-variant">
        <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-control-hover">
          <Folder className="h-8 w-8 opacity-40" />
        </div>
        <p className="text-body-sm font-medium">No assets yet</p>
        <p className="mt-1 max-w-xs text-center text-caption">
          Generated images and imported files will appear here.
        </p>
        <div className="mt-6 flex gap-4">
          <div className="flex flex-col items-center gap-1.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-control-hover">
              <Image className="h-4 w-4 text-on-surface-variant" />
            </div>
            <span className="text-caption text-on-surface-variant">Run a workflow</span>
          </div>
          <div className="flex flex-col items-center gap-1.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-control-hover">
              <Upload className="h-4 w-4 text-on-surface-variant" />
            </div>
            <span className="text-caption text-on-surface-variant">Import files</span>
          </div>
        </div>
      </div>
    </div>
  );
}