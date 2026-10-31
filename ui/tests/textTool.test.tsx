import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { toast } from "sonner";
import { fireEvent, render, screen, waitFor, within } from "./test-utils";
import { CollectionView } from "../src/components/layout/CollectionView";
import * as ipc from "../src/ipc";
import { mockOpList, mockOpInvoke, mockOpPreview } from "../src/ipc/mock";
import { useBoardStore } from "../src/store/board";
import { useProjectStore } from "../src/store/project";

const mockNavigate = vi.fn();
let mockParams = { id: "col_123" };

vi.mock("sonner", () => ({
  toast: {
    error: vi.fn(),
    success: vi.fn(),
  },
}));

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
  useRouter: () => ({ history: { canGoBack: () => false, back: vi.fn() } }),
}));

type CollectionReply = {
  id: string;
  name: string;
  version: number;
  slots: Array<{ id: string; asset_id: string }>;
};

function mockOperationsAvailable(): void {
  vi.spyOn(ipc, "opList").mockImplementation(() => mockOpList({ projectId: "default" }));
}

function mockCollection(
  reply: () => CollectionReply,
  handles: Record<string, unknown> = {},
): ReturnType<typeof vi.spyOn> {
  return vi.spyOn(ipc, "opInvoke").mockImplementation(async (_projectId, name, input) => {
    if (name === "collection.get") {
      return { step: {} as never, output: reply() };
    }
    if (name in handles) {
      const handler = handles[name];
      return {
        step: {} as never,
        output:
          typeof handler === "function" ? (handler as (i: unknown) => unknown)(input) : handler,
      };
    }
    return { step: {} as never, output: {} };
  });
}

async function openTool(): Promise<void> {
  await waitFor(() => {
    expect(screen.getByText("Working draft")).toBeInTheDocument();
  });
  fireEvent.click(screen.getByRole("button", { name: /text tool/i }));
  await waitFor(() => {
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByLabelText("Find")).toBeInTheDocument();
  });
}

function fillPair(find: string, replace: string): void {
  fireEvent.change(screen.getByLabelText("Find"), { target: { value: find } });
  fireEvent.change(screen.getByLabelText("Replace with"), { target: { value: replace } });
}

function applyButton(): HTMLElement {
  return screen.getByRole("button", { name: /^apply$/i });
}

async function waitForDiff(text: string): Promise<HTMLElement> {
  await waitFor(() => {
    expect(screen.getByTestId("text-tool-diff")).toHaveTextContent(text);
  });
  return screen.getByTestId("text-tool-diff");
}

describe("getPreviewUrl", () => {
  test("maps the host preview URI onto the platform asset base", () => {
    // jsdom has no Tauri bridge, so the plain-browser base is exercised.
    expect(ipc.getPreviewUrl("reimagine-asset://preview/key-1/out.md")).toBe(
      "reimagine-asset://localhost/preview/key-1/out.md",
    );
    expect(ipc.getPreviewUrl("reimagine-asset://preview/key-1/sub dir/out.md")).toBe(
      "reimagine-asset://localhost/preview/key-1/sub%20dir/out.md",
    );
    expect(ipc.getPreviewUrl("http://reimagine-asset.localhost/preview/key-2/out.md")).toBe(
      "reimagine-asset://localhost/preview/key-2/out.md",
    );
  });
});

describe("text replace tool", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockParams = { id: "col_123" };
    useProjectStore.setState({ activeProjectId: "default" });
    useBoardStore.getState().hydrate({
      id: "board-default",
      projectId: "default",
      version: 1,
      items: [],
    });
    vi.spyOn(ipc, "readAssetPreview").mockResolvedValue({
      url: "asset-url",
      mediaType: "text/markdown",
      text: "Hello world",
    });
    vi.spyOn(ipc, "readAssetText").mockResolvedValue("Hello world\nSecond line");
    mockCollection(() => ({
      id: "col_123",
      name: "Working draft",
      version: 2,
      slots: [{ id: "s1", asset_id: "ast_1" }],
    }));
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("hides the tool when op_list has no text.replace", async () => {
    vi.spyOn(ipc, "opList").mockResolvedValue([]);

    render(<CollectionView />);
    await waitFor(() => {
      expect(screen.getByText("Working draft")).toBeInTheDocument();
    });
    expect(screen.queryByRole("button", { name: /text tool/i })).toBeNull();
  });

  test("hides the tool when text.replace is unhealthy", async () => {
    const operations = await mockOpList({ projectId: "default" });
    vi.spyOn(ipc, "opList").mockResolvedValue(
      operations.map((operation) =>
        operation.spec.name === "text.replace"
          ? {
              ...operation,
              health: {
                status: "unavailable",
                code: "TEXT/UNAVAILABLE",
                message: "mod stopped",
              },
            }
          : operation,
      ),
    );

    render(<CollectionView />);
    await waitFor(() => {
      expect(screen.getByText("Working draft")).toBeInTheDocument();
    });
    expect(screen.queryByRole("button", { name: /text tool/i })).toBeNull();
  });

  test("opens from a slot and reads the full original document", async () => {
    mockOperationsAvailable();

    render(<CollectionView />);
    await openTool();

    expect(screen.getByRole("heading", { name: "Replace text" })).toBeInTheDocument();
    await waitFor(() => {
      expect(within(screen.getByRole("dialog")).getByText(/Hello world/)).toBeInTheDocument();
    });
    expect(ipc.readAssetText).toHaveBeenCalledWith("default", "ast_1");
    expect(screen.getByLabelText("Case sensitive")).toBeChecked();
  });

  test("debounces edits, previews with a stable key and shows a line diff", async () => {
    mockOperationsAvailable();
    const previewSpy = vi
      .spyOn(ipc, "opPreview")
      .mockResolvedValue({ text: "Hello there\nSecond line", counts: [1] });

    render(<CollectionView />);
    await openTool();

    fireEvent.change(screen.getByLabelText("Find"), { target: { value: "w" } });
    fireEvent.change(screen.getByLabelText("Find"), { target: { value: "world" } });
    fireEvent.change(screen.getByLabelText("Replace with"), { target: { value: "there" } });

    await waitFor(() => {
      expect(previewSpy).toHaveBeenCalledTimes(1);
    });
    const [projectId, name, input, previewKey] = previewSpy.mock.calls[0];
    expect(projectId).toBe("default");
    expect(name).toBe("text.replace");
    expect(input).toEqual({
      doc: "ast_1",
      pairs: [
        {
          find: "world",
          replace: "there",
          mode: "literal",
          case_sensitive: true,
        },
      ],
    });
    expect(previewKey).toMatch(/^[A-Za-z0-9_-]{1,64}$/);

    const diff = await waitForDiff("Hello there");
    expect(within(diff).getByText(/Hello there/)).toHaveAttribute("data-diff", "added");
    expect(within(diff).getByText(/Hello world/)).toHaveAttribute("data-diff", "removed");
    expect(screen.getByText("Pair 1: 1 replacements")).toBeInTheDocument();
    await waitFor(() => {
      expect(applyButton()).toBeEnabled();
    });
  });

  test("shows a preview diagnostic and keeps apply disabled", async () => {
    mockOperationsAvailable();
    vi.spyOn(ipc, "opPreview").mockRejectedValue(new Error("invalid regex"));

    render(<CollectionView />);
    await openTool();

    fillPair("world", "there");

    await waitFor(() => {
      expect(screen.getByText("Could not preview the change: invalid regex")).toBeInTheDocument();
    });
    expect(applyButton()).toBeDisabled();
  });

  test("confirms with one text.replace and one collection.set_slot", async () => {
    mockOperationsAvailable();
    vi.spyOn(ipc, "opPreview").mockResolvedValue({
      text: "Hello there\nSecond line",
      counts: [1],
    });
    const opInvokeSpy = mockCollection(
      () => ({
        id: "col_123",
        name: "Working draft",
        version: 2,
        slots: [{ id: "s1", asset_id: "ast_1" }],
      }),
      {
        "text.replace": { doc: "ast_new", counts: [1] },
        "collection.set_slot": { version: 3 },
      },
    );

    render(<CollectionView />);
    await openTool();
    fillPair("world", "there");

    await waitForDiff("Hello there");
    await waitFor(() => {
      expect(applyButton()).toBeEnabled();
    });
    fireEvent.click(applyButton());

    await waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });

    const names = opInvokeSpy.mock.calls.map(([, name]) => name);
    expect(names.filter((name) => name === "text.replace")).toHaveLength(1);
    expect(names.filter((name) => name === "collection.set_slot")).toHaveLength(1);
    expect(opInvokeSpy).toHaveBeenCalledWith("default", "text.replace", {
      doc: "ast_1",
      pairs: [
        {
          find: "world",
          replace: "there",
          mode: "literal",
          case_sensitive: true,
        },
      ],
    });
    expect(opInvokeSpy).toHaveBeenCalledWith("default", "collection.set_slot", {
      collection_id: "col_123",
      slot_id: "s1",
      asset: "ast_new",
    });
    // One read on mount and one guard read before the write.
    expect(names.filter((name) => name === "collection.get").length).toBeGreaterThanOrEqual(2);
    expect(toast.success).toHaveBeenCalledWith("Text replaced");
  });

  test("refuses to write when the slot moved since the preview", async () => {
    mockOperationsAvailable();
    vi.spyOn(ipc, "opPreview").mockResolvedValue({
      text: "Hello there\nSecond line",
      counts: [1],
    });
    let slotAsset = "ast_1";
    const opInvokeSpy = mockCollection(() => ({
      id: "col_123",
      name: "Working draft",
      version: 2,
      slots: [{ id: "s1", asset_id: slotAsset }],
    }));

    render(<CollectionView />);
    await openTool();
    fillPair("world", "there");
    await waitForDiff("Hello there");

    slotAsset = "ast_other";
    await waitFor(() => {
      expect(applyButton()).toBeEnabled();
    });
    fireEvent.click(applyButton());

    await waitFor(() => {
      expect(
        screen.getByText("The slot now points at a different document. Nothing was written."),
      ).toBeInTheDocument();
    });
    const names = opInvokeSpy.mock.calls.map(([, name]) => name);
    expect(names).not.toContain("text.replace");
    expect(names).not.toContain("collection.set_slot");
  });

  test("reads a preview document URI through the asset protocol", async () => {
    mockOperationsAvailable();
    vi.spyOn(ipc, "opPreview").mockResolvedValue({
      doc: "reimagine-asset://preview/abc123/out.md",
      counts: [2],
    });
    const fetchSpy = vi.fn(async (_url: string) => {
      return new Response("Hello there\nSecond line", {
        status: 200,
        headers: { "content-type": "text/markdown" },
      });
    });
    globalThis.fetch = fetchSpy as unknown as typeof fetch;

    render(<CollectionView />);
    await openTool();
    fillPair("world", "there");

    await waitForDiff("Hello there");
    expect(fetchSpy).toHaveBeenCalledWith("reimagine-asset://localhost/preview/abc123/out.md");
  });

  test("reports inline output instead of writing an asset", async () => {
    mockOperationsAvailable();
    vi.spyOn(ipc, "opPreview").mockResolvedValue({
      text: "Hello there\nSecond line",
      counts: [1],
    });
    const opInvokeSpy = mockCollection(
      () => ({
        id: "col_123",
        name: "Working draft",
        version: 2,
        slots: [{ id: "s1", asset_id: "ast_1" }],
      }),
      { "text.replace": { text: "Hello there\nSecond line", counts: [1] } },
    );

    render(<CollectionView />);
    await openTool();
    fillPair("world", "there");
    await waitForDiff("Hello there");
    await waitFor(() => {
      expect(applyButton()).toBeEnabled();
    });
    fireEvent.click(applyButton());

    await waitFor(() => {
      expect(
        screen.getByText(
          "The operation returned inline text instead of a document. Nothing was written.",
        ),
      ).toBeInTheDocument();
    });
    const names = opInvokeSpy.mock.calls.map(([, name]) => name);
    expect(names).not.toContain("collection.set_slot");
  });
});

describe("text replace mocks", () => {
  test("preview applies pairs in order and invoke writes an asset", async () => {
    const preview = (await mockOpPreview({
      projectId: "default",
      name: "text.replace",
      input: {
        doc: "ast_demo_blog_v2",
        pairs: [{ find: "Blog", replace: "Journal", mode: "literal" }],
      },
      previewKey: "pk_test",
    })) as { text: string; counts: number[] };
    expect(preview.counts).toEqual([1]);
    expect(preview.text).toContain("Journal v2");

    const invoked = await mockOpInvoke({
      projectId: "default",
      name: "text.replace",
      input: {
        doc: "ast_demo_blog_v2",
        pairs: [
          { find: "Blog", replace: "Journal", mode: "literal" },
          { find: "outline", replace: "summary", mode: "word", case_sensitive: false },
        ],
      },
    });
    const output = invoked.output as { doc: string; counts: number[] };
    expect(output.counts).toEqual([1, 1]);
    expect(output.doc).toMatch(/^ast_/);
  });
});
