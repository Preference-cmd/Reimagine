import { describe, test, expect, vi } from "vitest";
import { render, screen, fireEvent } from "./test-utils";
import { SidebarNav } from "../src/components/layout/SidebarNav";

const mockSetActive = vi.fn();
const mockSetProjectsExpanded = vi.fn();
const mockSetActiveProjectId = vi.fn();

vi.mock("../src/store/uiStore", () => ({
  useUIStore: Object.assign(
    vi.fn((selector?: (state: any) => any) => {
      const state = {
        activeSidebarSection: "chat",
        setActiveSidebarSection: mockSetActive,
        projectsExpanded: true,
        setProjectsExpanded: mockSetProjectsExpanded,
        activeProjectId: null,
        setActiveProjectId: mockSetActiveProjectId,
      };
      return selector ? selector(state) : state;
    }),
    {
      getState: () => ({
        activeSidebarSection: "chat",
        setActiveSidebarSection: mockSetActive,
        projectsExpanded: true,
        setProjectsExpanded: mockSetProjectsExpanded,
        activeProjectId: null,
        setActiveProjectId: mockSetActiveProjectId,
      }),
    },
  ),
}));

vi.mock("../src/store/workflow", () => ({
  useWorkflowStore: vi.fn(() => "Untitled"),
}));

vi.mock("$paraglide/messages", () => ({
  "sidebar.newTask": () => "New Task",
  "sidebar.pullRequests": () => "Pull Requests",
  "sidebar.scheduled": () => "Scheduled",
  "sidebar.plugins": () => "Plugins",
  "sidebar.projects": () => "Projects",
  "sidebar.tasks": () => "Tasks",
  "sidebar.noTasks": () => "No tasks",
}));

describe("SidebarNav", () => {
  test("renders all primary navigation items", () => {
    render(<SidebarNav />);
    expect(screen.getByText("New Task")).toBeInTheDocument();
    expect(screen.getByText("Pull Requests")).toBeInTheDocument();
    expect(screen.getByText("Scheduled")).toBeInTheDocument();
    expect(screen.getByText("Plugins")).toBeInTheDocument();
  });

  test("renders navigation with correct aria label", () => {
    render(<SidebarNav />);
    expect(screen.getByLabelText("Sidebar navigation")).toBeInTheDocument();
  });

  test("calls setActiveSidebarSection on click and clears project selection", () => {
    render(<SidebarNav />);
    mockSetActive.mockClear();
    mockSetActiveProjectId.mockClear();

    fireEvent.click(screen.getByText("Scheduled"));
    expect(mockSetActive).toHaveBeenCalledWith("scheduled");
    expect(mockSetActiveProjectId).toHaveBeenCalledWith(null);
  });
});
