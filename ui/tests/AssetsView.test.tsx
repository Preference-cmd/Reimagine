import { describe, test, expect } from "vitest";
import { render, screen } from "./test-utils";
import { AssetsView } from "../src/components/layout/AssetsView";

describe("AssetsView", () => {
  test("renders the page title", () => {
    render(<AssetsView />);
    expect(screen.getByText("Assets")).toBeInTheDocument();
  });

  test("shows empty state message", () => {
    render(<AssetsView />);
    expect(screen.getByText("No assets yet")).toBeInTheDocument();
  });

  test("shows empty state description", () => {
    render(<AssetsView />);
    expect(
      screen.getByText("Generated images and imported files will appear here."),
    ).toBeInTheDocument();
  });

  test("shows action hints", () => {
    render(<AssetsView />);
    expect(screen.getByText("Run a workflow")).toBeInTheDocument();
    expect(screen.getByText("Import files")).toBeInTheDocument();
  });
});
