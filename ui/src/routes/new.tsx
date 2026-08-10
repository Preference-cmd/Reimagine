import { createFileRoute } from "@tanstack/react-router";
import { WelcomeScreen } from "@/components/layout/WelcomeScreen";

function NewView() {
  return <WelcomeScreen />;
}

export const Route = createFileRoute("/new")({
  component: NewView,
});
