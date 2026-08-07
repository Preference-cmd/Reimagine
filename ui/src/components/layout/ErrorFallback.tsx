export function ErrorFallback({
  error,
  resetErrorBoundary,
}: {
  error: unknown;
  resetErrorBoundary: () => void;
}) {
  const message = error instanceof Error ? error.message : String(error);

  return (
    <div className="flex h-full items-center justify-center p-8">
      <div className="text-center">
        <h2 className="mb-2 text-body-lg font-semibold text-on-surface">Something went wrong</h2>
        <p className="mb-4 text-caption text-on-surface-variant">{message}</p>
        <button
          onClick={resetErrorBoundary}
          className="rounded-lg bg-primary px-4 py-2 text-body-sm font-medium text-on-primary transition-colors hover:bg-primary/90"
        >
          Try again
        </button>
      </div>
    </div>
  );
}
