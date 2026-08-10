import { createFileRoute } from "@tanstack/react-router";
import { AssetsView } from "@/components/layout/AssetsView";

export const Route = createFileRoute("/assets")({
  component: AssetsView,
});
