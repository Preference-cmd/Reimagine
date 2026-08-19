import { describe, test, expect, vi } from "vitest";
import { render, screen } from "./test-utils";
import { SidebarNav } from "../src/components/layout/SidebarNav";

vi.mock("@tanstack/react-router", () => ({
  useLocation: () => ({ pathname: "/workflows" }),
  Link: ({
    to,
    children,
    className,
    ...props
  }: {
    to: string;
    children: React.ReactNode;
    className?: string;
    [key: string]: unknown;
  }) => (
    <a href={to} className={className} {...props}>
      {children}
    </a>
  ),
}));

vi.mock("$paraglide/messages", () => ({
  "sidebar.new": () => "New",
  "sidebar.workflows": () => "Workflows",
  "sidebar.models": () => "Models",
  "sidebar.runs": () => "Runs",
  "sidebar.assets": () => "Assets",
}));

describe("SidebarNav", () => {
  test("renders all primary navigation items", () => {
    render(<SidebarNav />);
    expect(screen.getByText("New")).toBeInTheDocument();
    expect(screen.getByText("Models")).toBeInTheDocument();
    expect(screen.getByText("Runs")).toBeInTheDocument();
    expect(screen.getByText("Assets")).toBeInTheDocument();
  });

  test("renders navigation with correct aria label", () => {
    render(<SidebarNav />);
    expect(screen.getByLabelText("Sidebar navigation")).toBeInTheDocument();
  });

  test("renders links with correct hrefs", () => {
    render(<SidebarNav />);
    const links = screen.getAllByRole("link");
    const hrefs = links.map((link) => link.getAttribute("href"));
    expect(hrefs).toContain("/new");
    expect(hrefs).toContain("/models");
    expect(hrefs).toContain("/runs");
    expect(hrefs).toContain("/assets");
  });
});
