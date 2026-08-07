import { describe, test, expect, vi } from "vitest";
import { render, screen, fireEvent } from "./test-utils";
import { SidebarNav } from "../src/components/layout/SidebarNav";

// Mock the uiStore
vi.mock("@/store/uiStore", () => ({
  useUIStore: Object.assign(
    vi.fn((selector?: (state: any) => any) => {
      const state = {
        activeSidebarSection: "workflows",
        setActiveSidebarSection: vi.fn(),
      };
      return selector ? selector(state) : state;
    }),
    { getState: () => ({ activeSidebarSection: "workflows", setActiveSidebarSection: vi.fn() }) },
  ),
}));

describe("SidebarNav", () => {
  test("renders all navigation items", () => {
    render(<SidebarNav />);
    expect(screen.getByText("Workflows")).toBeInTheDocument();
    expect(screen.getByText("Models")).toBeInTheDocument();
    expect(screen.getByText("Runs")).toBeInTheDocument();
    expect(screen.getByText("Assets")).toBeInTheDocument();
    expect(screen.getByText("Settings")).toBeInTheDocument();
  });

  test("renders navigation with correct aria label", () => {
    render(<SidebarNav />);
    expect(screen.getByLabelText("Sidebar navigation")).toBeInTheDocument();
  });

  test("calls setActiveSidebarSection on click", async () => {
    const setActiveSidebarSection = vi.fn();
    const mockUseUIStore = vi.mocked((await import("@/store/uiStore")).useUIStore);
    mockUseUIStore.mockImplementation((selector?: (state: any) => any) => {
      const state = {
        activeSidebarSection: "workflows",
        setActiveSidebarSection,
      };
      return selector ? selector(state) : state;
    });

    render(<SidebarNav />);

    fireEvent.click(screen.getByText("Models"));
    expect(setActiveSidebarSection).toHaveBeenCalledWith("models");
  });
});
