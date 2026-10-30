import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { act, render, screen, fireEvent, waitFor } from "./test-utils";
import { CollectionView } from "../src/components/layout/CollectionView";
import * as ipc from "../src/ipc";
import { useBoardStore } from "../src/store/board";
import { useProjectStore } from "../src/store/project";
import * as artifacts from "../src/lib/artifacts";
import { mockOpInvoke } from "../src/ipc/mock";

const mockNavigate = vi.fn();
const mockHistory = {
  canGoBack: vi.fn(() => false),
  back: vi.fn(),
};
let mockParams = { id: "col_123" };
useProjectStore.setState({ activeProjectId: "default" });

vi.mock("@tanstack/react-router", () => ({
  Link: ({
    to,
    children,
    ...props
  }: {
    to: string;
    children: React.ReactNode;
    [key: string]: unknown;
  }) => (
    <a href={to} {...props}>
      {children}
    </a>
  ),
  useNavigate: () => mockNavigate,
  useParams: () => mockParams,
  useRouter: () => ({ history: mockHistory }),
}));

describe("Collections and Dynamic Workflow UI", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockParams = { id: "col_123" };
    mockHistory.canGoBack.mockReturnValue(false);
    useProjectStore.setState({ activeProjectId: "default" });
    useBoardStore.getState().hydrate({
      id: "board-default",
      projectId: "default",
      version: 1,
      items: [],
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("the view renders slots with media type and text snippet via asset URL", async () => {
    vi.spyOn(ipc, "opInvoke").mockImplementation(async (_proj, name, _input) => {
      if (name === "collection.get") {
        return {
          step: {} as unknown as never,
          output: {
            id: "col_123",
            name: "Research Papers",
            version: 3,
            slots: [
              { id: "s1", asset_id: "ast_1" },
              { id: "s2", asset_id: "ast_2" },
            ],
          },
        };
      }
      return { step: {} as unknown as never, output: {} };
    });

    // Mock fetch for asset URL requests
    globalThis.fetch = vi.fn().mockImplementation(async (url: string) => {
      if (url.includes("ast_1")) {
        return new Response("Title: Transformer Attention\nLine 2: Abstract\nLine 3: Details", {
          status: 200,
          headers: { "content-type": "text/markdown" },
        });
      }
      return new Response("Plain text file\nSecond line", {
        status: 200,
        headers: { "content-type": "text/plain" },
      });
    });

    render(<CollectionView />);

    await waitFor(() => {
      expect(screen.getByText("Research Papers")).toBeInTheDocument();
      expect(screen.getByText(/2 slots/i)).toBeInTheDocument();
    });

    expect(screen.getByText("Slot: s1")).toBeInTheDocument();
    expect(screen.getByText("Slot: s2")).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText("text/markdown")).toBeInTheDocument();
      expect(screen.getByText("text/plain")).toBeInTheDocument();
      expect(screen.getByText(/Title: Transformer Attention/)).toBeInTheDocument();
    });
  });

  test("undo calls collection.undo and redo calls collection.redo", async () => {
    const opInvokeSpy = vi
      .spyOn(ipc, "opInvoke")
      .mockImplementation(async (_proj, name, _input) => {
        if (name === "collection.get") {
          return {
            step: {} as unknown as never,
            output: {
              id: "col_123",
              name: "My Collection",
              version: 2,
              slots: [{ id: "s1", asset_id: "ast_1" }],
            },
          };
        }
        if (name === "collection.undo") {
          return {
            step: {} as unknown as never,
            output: { applied: true, version: 1 },
          };
        }
        if (name === "collection.redo") {
          return {
            step: {} as unknown as never,
            output: { applied: true, version: 2 },
          };
        }
        return { step: {} as unknown as never, output: {} };
      });

    render(<CollectionView />);

    await waitFor(() => {
      expect(screen.getByText("My Collection")).toBeInTheDocument();
    });

    const undoBtn = screen.getByRole("button", { name: /undo/i });
    fireEvent.click(undoBtn);

    await waitFor(() => {
      expect(opInvokeSpy).toHaveBeenCalledWith("default", "collection.undo", {
        collection_id: "col_123",
      });
    });

    const redoBtn = screen.getByRole("button", { name: /redo/i });
    fireEvent.click(redoBtn);

    await waitFor(() => {
      expect(opInvokeSpy).toHaveBeenCalledWith("default", "collection.redo", {
        collection_id: "col_123",
      });
    });
  });

  test("a collection.changed event triggers a refetch", async () => {
    const eventCallbacks: Array<(event: unknown) => void> = [];
    vi.spyOn(ipc, "subscribeEvents").mockImplementation((cb) => {
      eventCallbacks.push(cb as unknown as (event: unknown) => void);
      return () => {
        const index = eventCallbacks.indexOf(cb as unknown as (event: unknown) => void);
        if (index >= 0) eventCallbacks.splice(index, 1);
      };
    });

    let getCalls = 0;
    vi.spyOn(ipc, "opInvoke").mockImplementation(async (_proj, name, _input) => {
      if (name === "collection.get") {
        getCalls += 1;
        return {
          step: {} as unknown as never,
          output: {
            id: "col_123",
            name: getCalls === 1 ? "Initial Name" : "Refetched Name",
            version: getCalls,
            slots: [{ id: "s1", asset_id: "ast_1" }],
          },
        };
      }
      return { step: {} as unknown as never, output: {} };
    });

    render(<CollectionView />);

    await waitFor(() => {
      expect(screen.getByText("Initial Name")).toBeInTheDocument();
    });
    expect(getCalls).toBe(1);

    // Emit collection.changed event for this collection
    expect(eventCallbacks.length).toBeGreaterThan(0);
    act(() => {
      for (const callback of [...eventCallbacks]) {
        callback({
          type: "document",
          kind: "collection.changed",
          projectId: "default",
          documentId: "col_123",
          version: 2,
        });
      }
    });

    await waitFor(() => {
      expect(screen.getByText("Refetched Name")).toBeInTheDocument();
    });
    expect(getCalls).toBe(2);

    // Events from other projects must be ignored
    act(() => {
      for (const callback of [...eventCallbacks]) {
        callback({
          type: "document",
          kind: "collection.changed",
          projectId: "other-project",
          documentId: "col_123",
          version: 3,
        });
      }
    });
    expect(getCalls).toBe(2);
  });

  test("slot run states follow the latest run and refresh on workflow events", async () => {
    const eventCallbacks: Array<(event: unknown) => void> = [];
    vi.spyOn(ipc, "subscribeEvents").mockImplementation((cb) => {
      eventCallbacks.push(cb as unknown as (event: unknown) => void);
      return () => {};
    });
    vi.spyOn(ipc, "readAssetPreview").mockResolvedValue({
      url: "asset-url",
      mediaType: "text/plain",
      text: "body",
    });

    let listCalls = 0;
    vi.spyOn(ipc, "opInvoke").mockImplementation(async (_proj, name, _input) => {
      if (name === "collection.get") {
        return {
          step: {} as unknown as never,
          output: {
            id: "col_123",
            name: "Run states",
            version: 1,
            slots: [
              { id: "s1", asset_id: "ast_1" },
              { id: "s2", asset_id: "ast_2" },
            ],
          },
        };
      }
      if (name === "dynamic_workflow.list_runs") {
        listCalls += 1;
        const done = listCalls > 1;
        return {
          step: {} as unknown as never,
          output: {
            runs: [
              {
                id: "run-1",
                project_id: "default",
                dynamic_workflow_id: "dwf-1",
                target: { collection_id: "col_123" },
                state: done ? "completed" : "running",
                version: listCalls,
                items: [
                  {
                    item_id: "s1",
                    slot_id: "s1",
                    state: done ? "succeeded" : "running",
                  },
                  { item_id: "s2", slot_id: "s2", state: "failed" },
                ],
                created_at: "2026-01-01T00:00:00Z",
              },
            ],
          },
        };
      }
      return { step: {} as unknown as never, output: {} };
    });

    render(<CollectionView />);

    await waitFor(() => {
      expect(screen.getByText("Running")).toBeInTheDocument();
      expect(screen.getByText("Failed")).toBeInTheDocument();
    });
    expect(listCalls).toBe(1);

    act(() => {
      for (const callback of [...eventCallbacks]) {
        callback({
          type: "document",
          kind: "dynamic_workflow.slot_changed",
          projectId: "default",
          documentId: "run-1",
          version: 2,
        });
      }
    });
    await waitFor(() => {
      expect(screen.getByText("Succeeded")).toBeInTheDocument();
    });
    expect(listCalls).toBe(2);

    act(() => {
      for (const callback of [...eventCallbacks]) {
        callback({
          type: "document",
          kind: "dynamic_workflow.run_changed",
          projectId: "default",
          documentId: "run-1",
          version: 3,
        });
      }
    });
    await waitFor(() => {
      expect(listCalls).toBe(3);
    });

    // Runs are started elsewhere; this view only reports state.
    expect(screen.queryByRole("button", { name: /run/i })).toBeNull();
  });

  test("back prefers router history and falls back to the library", async () => {
    vi.spyOn(ipc, "opInvoke").mockImplementation(async (_proj, name, _input) => {
      if (name === "collection.get") {
        return {
          step: {} as unknown as never,
          output: {
            id: "col_123",
            name: "History Collection",
            version: 1,
            slots: [],
          },
        };
      }
      return { step: {} as unknown as never, output: {} };
    });

    mockHistory.canGoBack.mockReturnValue(true);
    render(<CollectionView />);
    await waitFor(() => {
      expect(screen.getByText("History Collection")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: "Back" }));
    expect(mockHistory.back).toHaveBeenCalledTimes(1);
    expect(mockNavigate).not.toHaveBeenCalledWith({ to: "/library" });

    // Direct links have no history entry: fall back to the library.
    mockHistory.canGoBack.mockReturnValue(false);
    fireEvent.click(screen.getByRole("button", { name: "Back" }));
    expect(mockNavigate).toHaveBeenCalledWith({ to: "/library" });
  });

  test("switching project navigates back to /library", async () => {
    vi.spyOn(ipc, "opInvoke").mockImplementation(async (_proj, name, _input) => {
      if (name === "collection.get") {
        return {
          step: {} as unknown as never,
          output: {
            id: "col_123",
            name: "Collection A",
            version: 1,
            slots: [],
          },
        };
      }
      return { step: {} as unknown as never, output: {} };
    });

    render(<CollectionView />);
    await waitFor(() => {
      expect(screen.getByText("Collection A")).toBeInTheDocument();
    });

    // Switch project
    act(() => {
      useProjectStore.setState({ activeProjectId: "another-project" });
    });

    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith({ to: "/library" });
    });
  });

  test("slot shows text lines, an image, or the read error", async () => {
    vi.spyOn(ipc, "opInvoke").mockImplementation(async (_proj, name, _input) => {
      if (name === "collection.get") {
        return {
          step: {} as unknown as never,
          output: {
            id: "col_123",
            name: "Collection Slots",
            version: 1,
            slots: [
              { id: "s1", asset_id: "ast_text" },
              { id: "s2", asset_id: "ast_img" },
              { id: "s3", asset_id: "ast_err" },
            ],
          },
        };
      }
      return { step: {} as unknown as never, output: {} };
    });
    vi.spyOn(ipc, "readAssetPreview").mockImplementation(async (_proj, assetId) => {
      if (assetId === "ast_text") {
        return { url: "u-text", mediaType: "text/markdown", text: "# Title\nbody" };
      }
      if (assetId === "ast_img") {
        return { url: "u-img", mediaType: "image/png", text: null };
      }
      throw new Error("HTTP 404 reading asset ast_err");
    });

    render(<CollectionView />);
    await waitFor(() => {
      expect(screen.getByText(/# Title/)).toBeInTheDocument();
    });
    expect(screen.getByAltText("s2")).toHaveAttribute("src", "u-img");
    expect(screen.getByText("HTTP 404 reading asset ast_err")).toBeInTheDocument();
  });

  test("asset URLs keep the project and asset as separate path segments", () => {
    vi.spyOn(artifacts, "tauriAvailable").mockReturnValue(false);
    expect(ipc.getAssetUrl("proj-1", "asset-1")).toBe("reimagine-asset://localhost/proj-1/asset-1");
  });

  test("mock error codes match backend collection_ops", async () => {
    await expect(
      mockOpInvoke({
        projectId: "default",
        name: "collection.get",
        input: { collection_id: "nonexistent" },
      }),
    ).rejects.toThrow("COLLECTION/CALLER_UNKNOWN_COLLECTION");

    await expect(
      mockOpInvoke({
        projectId: "default",
        name: "collection.set_slot",
        input: { collection_id: "nonexistent", slot_id: "s1", asset: "a1" },
      }),
    ).rejects.toThrow("COLLECTION/CALLER_UNKNOWN_COLLECTION");

    const created = await mockOpInvoke({
      projectId: "default",
      name: "collection.create",
      input: { name: "C", assets: ["a1"] },
    });
    const collectionId = (created.output as { collection_id: string }).collection_id;
    await expect(
      mockOpInvoke({
        projectId: "default",
        name: "collection.set_slot",
        input: { collection_id: collectionId, slot_id: "slot-9", asset: "a1" },
      }),
    ).rejects.toThrow("COLLECTION/CALLER_UNKNOWN_SLOT");
  });
});
