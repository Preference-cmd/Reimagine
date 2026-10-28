import { createFileRoute } from "@tanstack/react-router";
import { CollectionView } from "@/components/layout/CollectionView";

export const Route = createFileRoute("/collections/$id")({
  component: CollectionView,
});
