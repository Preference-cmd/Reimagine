import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import type { ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "./test-utils";
import { LibraryView } from "../src/components/layout/LibraryView";
import * as ipc from "../src/ipc";
import type {
  AssetSummary,
  CollectionSummary,
  DynamicWorkflowRun,
  DynamicWorkflowSummary,
} from "../src/ipc/schemas";
import {
  buildLibraryRows,
  countLibraryRows,
  filterLibraryRows,
  formatBytes,
} from "../src/lib/library";
import { useProjectStore } from "../src/store/project";
import { useUIStore } from "../src/store/uiStore";

const mockNavigate = vi.fn();
let mockSearch: { type?: string } = {};

vi.mock("@tanstack/react-router", () => ({
  Link: ({ to, children, ...props }: { to: string; children: ReactNode }) => (
    <a href={to} {...props}>
      {children}
    </a>
  ),
  useNavigate: () => mockNavigate,
  useSearch: () => mockSearch,
}));

const collections: CollectionSummary[] = [
  { id: "col_alpha", name: "Alpha Collection", version: 3, slot_count: 4 },
  { id: "col_beta", name: "Beta Collection", version: 1, slot_count: 2 },
];

const workflows: DynamicWorkflowSummary[] = [
  { id: "dwf_cleanup", name: "Draft cleanup", version: 2, step_count: 3 },
  { id: "dwf_archive", name: "Archive sweep", version: 1, step_count: 2 },
];

const runs: DynamicWorkflowRun[] = [
  {
    id: "run_2",
    project_id: "default",
    dynamic_workflow_id: "dwf_cleanup",
    target: {},
    state: "completed",
    version: 1,
    items: [
      { item_id: "i1", slot_id: "slot-1", state: "succeeded" },
      { item_id: "i2", slot_id: "slot-2", state: "succeeded" },
      { item_id: "i3", slot_id: "slot-3", state: "failed" },
    ],
    collection_id: "col_alpha",
    created_at: "2026-10-29T10:00:00Z",
  },
  {
    id: "run_1",
    project_id: "default",
    dynamic_workflow_id: "dwf_missing",
    target: {},
    state: "running",
    version: 1,
    items: [{ item_id: "i1", slot_id: "slot-1", state: "running" }],
    created_at: "2026-10-29T09:00:00Z",
  },
];

const assets: AssetSummary[] = [
  { id: "ast_abc", media_type: "image/png", size: 2048, origin: "step-1" },
  { id: "ast_def", media_type: "text/markdown", size: 1500, origin: "step-2" },
];

/** `op_invoke` output by operation name, used by the mocked IPC. */
function opOutput(name: string): unknown {
  switch (name) {
    case "collection.list":
      return { collections };
    case "dynamic_workflow.list":
      return { workflows };
    case "dynamic_workflow.list_runs":
      return { runs };
    case "asset.list":
      return { assets };
    default:
      return {};
  }
}

function mockOpInvoke(output: (name: string) => unknown = opOutput) {
  return vi
    .spyOn(ipc, "opInvoke")
    .mockImplementation(async (_projectId, name) => ({ step: {}, output: output(name) }));
}

function renderLibrary() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: Infinity } },
  });
  return render(
    <QueryClientProvider client={client}>
      <LibraryView />
    </QueryClientProvider>,
  );
}

async function expectRows(count: number) {
  await waitFor(() => {
    expect(screen.getAllByTestId("library-row")).toHaveLength(count);
  });
}

describe("library row helpers", () => {
  const rows = buildLibraryRows({ collections, workflows, runs, assets });

  test("sorts by type then name and formats each detail", () => {
    expect(rows.map((row) => `${row.type}:${row.name}`)).toEqual([
      "collection:Alpha Collection",
      "collection:Beta Collection",
      "workflow:Archive sweep",
      "workflow:Draft cleanup",
      "run:Draft cleanup",
      "run:dwf_missing",
      "asset:ast_abc",
      "asset:ast_def",
    ]);
    expect(rows[0].detail).toBe("4 slots · v3");
    expect(rows[2].detail).toBe("2 steps · v1");
    expect(rows[4].detail).toBe("completed · 2/3");
    expect(rows[4].meta).toBe("2026-10-29T10:00:00Z");
    expect(rows[5].detail).toBe("running · 0/1");
    expect(rows[6].detail).toBe("image/png · 2 KB");
    expect(rows[7].detail).toBe("text/markdown · 1.5 KB");
  });

  test("filters by type and by a trimmed, case-insensitive name or detail query", () => {
    expect(filterLibraryRows(rows, "run", "")).toHaveLength(2);
    expect(filterLibraryRows(rows, "all", "  DRAFT ").map((row) => row.type)).toEqual([
      "workflow",
      "run",
    ]);
    expect(filterLibraryRows(rows, "all", "image/png").map((row) => row.id)).toEqual(["ast_abc"]);
    expect(filterLibraryRows(rows, "collection", "draft")).toHaveLength(0);
    expect(filterLibraryRows(rows, "all", "nothing-here")).toHaveLength(0);
  });

  test("counts every row per filter", () => {
    expect(countLibraryRows(rows)).toEqual({
      all: 8,
      collection: 2,
      workflow: 2,
      run: 2,
      asset: 2,
    });
  });

  test("formats byte sizes", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1500)).toBe("1.5 KB");
    expect(formatBytes(2048)).toBe("2 KB");
    expect(formatBytes(150 * 1024)).toBe("150 KB");
  });
});

describe("LibraryView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockSearch = {};
    useProjectStore.setState({ activeProjectId: "default" });
    useUIStore.setState({ sessionPanelOpen: false });
    mockOpInvoke();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("renders the four library kinds with type counts and joined run names", async () => {
    renderLibrary();
    await expectRows(8);

    expect(screen.getByText("Alpha Collection")).toBeInTheDocument();
    expect(screen.getByText("4 slots · v3")).toBeInTheDocument();
    expect(screen.getByText("2 steps · v1")).toBeInTheDocument();
    expect(screen.getByText("Completed · 2/3")).toBeInTheDocument();
    expect(screen.getByText("image/png · 2 KB")).toBeInTheDocument();
    // The run joins the workflow name; both rows carry "Draft cleanup".
    expect(screen.getAllByText("Draft cleanup")).toHaveLength(2);
    expect(screen.getByText("dwf_missing")).toBeInTheDocument();

    const chipCounts: Array<[string, string]> = [
      ["all", "8"],
      ["collection", "2"],
      ["workflow", "2"],
      ["run", "2"],
      ["asset", "2"],
    ];
    for (const [chip, count] of chipCounts) {
      expect(
        within(screen.getByTestId(`library-chip-${chip}`)).getByText(count),
      ).toBeInTheDocument();
    }
    expect(screen.getByTestId("library-chip-all")).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByTestId("library-import")).toBeDisabled();
    expect(ipc.opInvoke).toHaveBeenCalledWith("default", "collection.list", {});
    expect(ipc.opInvoke).toHaveBeenCalledWith("default", "dynamic_workflow.list", {});
    expect(ipc.opInvoke).toHaveBeenCalledWith("default", "asset.list", {});
    expect(ipc.opInvoke).toHaveBeenCalledWith("default", "dynamic_workflow.list_runs", {
      limit: 50,
    });
  });

  test("filters by name and details as the search is typed", async () => {
    renderLibrary();
    await expectRows(8);

    fireEvent.change(screen.getByTestId("library-search"), { target: { value: "  DRAFT " } });
    await expectRows(2);
    expect(screen.getAllByText("Draft cleanup")).toHaveLength(2);

    fireEvent.change(screen.getByTestId("library-search"), { target: { value: "image/png" } });
    await expectRows(1);
    expect(screen.getByText("ast_abc")).toBeInTheDocument();

    fireEvent.change(screen.getByTestId("library-search"), { target: { value: "" } });
    await expectRows(8);
  });

  test("type chips filter rows, update the URL and keep counts at all", async () => {
    renderLibrary();
    await expectRows(8);

    fireEvent.click(screen.getByTestId("library-chip-run"));
    await expectRows(2);
    expect(screen.getByTestId("library-chip-run")).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByTestId("library-chip-all")).toHaveAttribute("aria-pressed", "false");
    expect(screen.getAllByText("Run")).toHaveLength(2);
    expect(within(screen.getByTestId("library-chip-all")).getByText("8")).toBeInTheDocument();
    expect(mockNavigate).toHaveBeenCalledWith({
      to: "/library",
      search: { type: "run" },
      replace: true,
    });

    fireEvent.click(screen.getByTestId("library-chip-all"));
    await expectRows(8);
    expect(mockNavigate).toHaveBeenLastCalledWith({
      to: "/library",
      search: { type: undefined },
      replace: true,
    });
  });

  test("honors ?type=run on mount without navigating", async () => {
    mockSearch = { type: "run" };
    renderLibrary();
    await expectRows(2);

    expect(screen.getByTestId("library-chip-run")).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByTestId("library-chip-all")).toHaveAttribute("aria-pressed", "false");
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  test("shows the all-empty state when every list is empty", async () => {
    mockOpInvoke((name) => {
      if (name === "collection.list") return { collections: [] };
      if (name === "dynamic_workflow.list") return { workflows: [] };
      if (name === "dynamic_workflow.list_runs") return { runs: [] };
      return { assets: [] };
    });

    renderLibrary();
    await waitFor(() => {
      expect(screen.getByText("The library is empty")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("library-table")).toBeNull();
  });

  test("shows the filtered-empty state when no row matches the search", async () => {
    renderLibrary();
    await expectRows(8);

    fireEvent.change(screen.getByTestId("library-search"), { target: { value: "zzz" } });
    await waitFor(() => {
      expect(screen.getByText("No matching items")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("library-table")).toBeNull();
  });

  test("shows an error with a retry that reloads the lists", async () => {
    let failing = true;
    mockOpInvoke((name) => {
      if (name === "collection.list" && failing) throw new Error("host offline");
      return opOutput(name);
    });

    renderLibrary();
    await waitFor(() => {
      expect(screen.getByText("Could not load the library")).toBeInTheDocument();
    });
    expect(screen.getByText("host offline")).toBeInTheDocument();

    failing = false;
    fireEvent.click(screen.getByTestId("library-retry"));
    await expectRows(8);
  });
});
