import { render, type RenderOptions } from "@testing-library/react";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ErrorBoundary } from "react-error-boundary";

function AllProviders({ children }: { children: React.ReactNode }) {
  return (
    <TooltipProvider>
      <ErrorBoundary fallbackRender={() => <div>Error</div>}>{children}</ErrorBoundary>
    </TooltipProvider>
  );
}

function customRender(ui: React.ReactElement, options?: Omit<RenderOptions, "wrapper">) {
  return render(ui, { wrapper: AllProviders, ...options });
}

export * from "@testing-library/react";
export { customRender as render };
