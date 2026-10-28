import { createFileRoute } from "@tanstack/react-router";
import { CollectionsView } from "@/components/layout/CollectionsView";

export const Route = createFileRoute("/collections/")({
  component: CollectionsView,
});
