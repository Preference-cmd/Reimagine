import { describe, test, expect, vi } from "vitest";
import { render, screen, fireEvent } from "./test-utils";
import { SidebarNav } from "../src/components/layout/SidebarNav";

const mockSetActive = vi.fn();

vi.mock("../src/store/uiStore", () => ({
  useUIStore: Object.assign(
    vi.fn((selector?: (state: any) => any) => {
      const state = {
        activeSidebarSection: "workflows",
        setActiveSidebarSection: mockSetActive,
      };
      return selector ? selector(state) : state;
    }),
    {
      getState: () => ({
        activeSidebarSection: "workflows",
        setActiveSidebarSection: mockSetActive,
      }),
    },
  ),
}));

vi.mock("$paraglide/messages", () => ({
  "sidebar.workflows": () => "Workflows",
  "sidebar.models": () => "Models",
  "sidebar.runs": () => "Runs",
  "sidebar.assets": () => "Assets",
}));

describe("SidebarNav", () => {
  test("renders all navigation items", () => {
    render(<SidebarNav />);
    expect(screen.getByText("Workflows")).toBeInTheDocument();
    expect(screen.getByText("Models")).toBeInTheDocument();
    expect(screen.getByText("Runs")).toBeInTheDocument();
    expect(screen.getByText("Assets")).toBeInTheDocument();
  });

  test("renders navigation with correct aria label", () => {
    render(<SidebarNav />);
    expect(screen.getByLabelText("Sidebar navigation")).toBeInTheDocument();
  });

  test("calls setActiveSidebarSection on click", () => {
    render(<SidebarNav />);
    mockSetActive.mockClear();

    fireEvent.click(screen.getByText("Models"));
    expect(mockSetActive).toHaveBeenCalledWith("models");
  });
});
