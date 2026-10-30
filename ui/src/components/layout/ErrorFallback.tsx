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
        <h2 className="mb-2 text-body-lg font-semibold text-fg">Something went wrong</h2>
        <p className="mb-4 text-caption text-fg3">{message}</p>
        <button
          onClick={resetErrorBoundary}
          className="rounded-lg bg-accent px-4 py-2 text-body-sm font-medium text-on-accent transition-colors hover:bg-accent-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/30"
        >
          Try again
        </button>
      </div>
    </div>
  );
}
