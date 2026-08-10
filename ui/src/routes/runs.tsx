import { createFileRoute } from "@tanstack/react-router";
import { RunsView } from "@/components/layout/RunsView";

export const Route = createFileRoute("/runs")({
  component: RunsView,
});
