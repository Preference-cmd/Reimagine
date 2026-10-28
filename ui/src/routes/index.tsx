import { createFileRoute } from "@tanstack/react-router";
import { BoardCanvas } from "@/board/BoardCanvas";

function IndexView() {
  return <BoardCanvas />;
}

export const Route = createFileRoute("/")({
  component: IndexView,
});
