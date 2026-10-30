import { createFileRoute } from "@tanstack/react-router";
import { LibraryView } from "@/components/layout/LibraryView";

export const Route = createFileRoute("/library")({
  validateSearch: (search: Record<string, unknown>): { type?: string } => ({
    type: typeof search.type === "string" ? search.type : undefined,
  }),
  component: LibraryView,
});
