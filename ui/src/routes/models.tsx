import { createFileRoute } from "@tanstack/react-router";
import { ModelsView } from "@/components/layout/ModelsView";

export const Route = createFileRoute("/models")({
  component: ModelsView,
});
