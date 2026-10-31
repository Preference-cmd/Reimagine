import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { toast } from "sonner";
import { act, fireEvent, render, screen, waitFor, within } from "./test-utils";
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
      const output =
        typeof handler === "function"
          ? await (handler as (i: unknown) => unknown | Promise<unknown>)(input)
          : handler;
      return { step: {} as never, output };
    }
    return { step: {} as never, output: {} };
  });
}

function textToolButton(): HTMLElement | null {
  return screen.queryByRole("button", { name: /text tool/i });
}

async function openTool(): Promise<void> {
  await waitFor(() => {
    expect(screen.getByText("Working draft")).toBeInTheDocument();
  });
  // The button also waits for the slot's media type, so the click must wait
  // for the preview read to report a textual asset.
  await waitFor(() => {
    expect(textToolButton()).toBeInTheDocument();
  });
  fireEvent.click(screen.getByRole("button", { name: /text tool/i }));
  await waitFor(() => {
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByLabelText("Find")).toBeInTheDocument();
  });
}

/** Let a resolved `op_list` reach the hook before asserting tool absence. */
async function settleOperations(): Promise<void> {
  await waitFor(() => {
    expect(ipc.opList).toHaveBeenCalled();
  });
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
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

  test("adds a cache-busting query and keeps encoding each segment", () => {
    expect(ipc.getPreviewUrl("reimagine-asset://preview/key-1/sub dir/out.md", 7)).toBe(
      "reimagine-asset://localhost/preview/key-1/sub%20dir/out.md?v=7",
    );
    // A query already on the host URI is replaced, not duplicated.
    expect(ipc.getPreviewUrl("reimagine-asset://preview/key-1/out.md?v=2", 9)).toBe(
      "reimagine-asset://localhost/preview/key-1/out.md?v=9",
    );
  });

  test("rejects dot segments and URIs without a preview path", () => {
    expect(() => ipc.getPreviewUrl("reimagine-asset://preview/../escape/out.md")).toThrow();
    expect(() => ipc.getPreviewUrl("reimagine-asset://preview/key-1/../out.md")).toThrow();
    expect(() => ipc.getPreviewUrl("reimagine-asset://preview/key-1/./out.md")).toThrow();
    expect(() => ipc.getPreviewUrl("reimagine-asset://key-1/out.md")).toThrow();
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
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  test("hides the tool when op_list has no text.replace", async () => {
    vi.spyOn(ipc, "opList").mockResolvedValue([]);

    render(<CollectionView />);
    await waitFor(() => {
      expect(screen.getByText("Working draft")).toBeInTheDocument();
    });
    await settleOperations();
    expect(textToolButton()).toBeNull();
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
    await settleOperations();
    expect(textToolButton()).toBeNull();
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

  test("does not preview before the debounce and reuses one preview key", async () => {
    mockOperationsAvailable();
    const previewSpy = vi
      .spyOn(ipc, "opPreview")
      .mockResolvedValue({ text: "Hello there\nSecond line", counts: [1] });

    render(<CollectionView />);
    await openTool();
    vi.useFakeTimers();
    try {
      fireEvent.change(screen.getByLabelText("Find"), { target: { value: "w" } });
      fireEvent.change(screen.getByLabelText("Find"), { target: { value: "world" } });
      fireEvent.change(screen.getByLabelText("Replace with"), { target: { value: "there" } });

      await act(async () => {
        await vi.advanceTimersByTimeAsync(274);
      });
      expect(previewSpy).not.toHaveBeenCalled();

      await act(async () => {
        await vi.advanceTimersByTimeAsync(1);
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      expect(previewSpy).toHaveBeenCalledTimes(1);

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

      const diff = screen.getByTestId("text-tool-diff");
      expect(within(diff).getByText(/Hello there/)).toHaveAttribute("data-diff", "added");
      expect(within(diff).getByText(/Hello world/)).toHaveAttribute("data-diff", "removed");
      expect(screen.getByText("Pair 1: 1 replacements")).toBeInTheDocument();
      expect(applyButton()).toBeEnabled();

      // Editing again restarts the debounce and previews through the same key.
      fireEvent.change(screen.getByLabelText("Replace with"), { target: { value: "there!" } });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(274);
      });
      expect(previewSpy).toHaveBeenCalledTimes(1);
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1);
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      expect(previewSpy).toHaveBeenCalledTimes(2);
      expect(previewSpy.mock.calls[1][3]).toBe(previewSpy.mock.calls[0][3]);
    } finally {
      vi.useRealTimers();
    }
  });

  test("does not preview while no pair has a find string", async () => {
    mockOperationsAvailable();
    const previewSpy = vi.spyOn(ipc, "opPreview").mockResolvedValue({ text: "", counts: [] });

    render(<CollectionView />);
    await openTool();
    vi.useFakeTimers();
    try {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1000);
      });
      expect(previewSpy).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });

  test("adds and removes pairs and sends only the pairs with a find string", async () => {
    mockOperationsAvailable();
    const previewSpy = vi
      .spyOn(ipc, "opPreview")
      .mockResolvedValue({ text: "Hello there\nSecond line", counts: [1, 1] });

    render(<CollectionView />);
    await openTool();
    vi.useFakeTimers();
    try {
      fireEvent.click(screen.getByRole("button", { name: "Add pair" }));
      const finds = screen.getAllByLabelText("Find");
      const replaces = screen.getAllByLabelText("Replace with");
      expect(finds).toHaveLength(2);

      fireEvent.change(finds[0], { target: { value: "world" } });
      fireEvent.change(replaces[0], { target: { value: "there" } });
      fireEvent.change(finds[1], { target: { value: "Second" } });
      fireEvent.change(replaces[1], { target: { value: "Next" } });

      await act(async () => {
        await vi.advanceTimersByTimeAsync(275);
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      const firstInput = previewSpy.mock.calls[0][2] as {
        pairs: Array<Record<string, unknown>>;
      };
      expect(firstInput.pairs).toHaveLength(2);

      fireEvent.click(screen.getAllByRole("button", { name: "Remove pair" })[1]);
      await act(async () => {
        await vi.advanceTimersByTimeAsync(275);
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      const secondInput = previewSpy.mock.calls[1][2] as {
        pairs: Array<Record<string, unknown>>;
      };
      expect(secondInput.pairs).toHaveLength(1);
      expect(secondInput.pairs[0].find).toBe("world");
    } finally {
      vi.useRealTimers();
    }
  });

  test("sends the selected replacement mode", async () => {
    mockOperationsAvailable();
    const previewSpy = vi
      .spyOn(ipc, "opPreview")
      .mockResolvedValue({ text: "Hello there\nSecond line", counts: [1] });

    render(<CollectionView />);
    await openTool();
    vi.useFakeTimers();
    try {
      fireEvent.change(screen.getByLabelText("Mode"), { target: { value: "regex" } });
      fireEvent.change(screen.getByLabelText("Find"), { target: { value: "w.rld" } });
      fireEvent.change(screen.getByLabelText("Replace with"), { target: { value: "there" } });

      await act(async () => {
        await vi.advanceTimersByTimeAsync(275);
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      const input = previewSpy.mock.calls[0][2] as { pairs: Array<Record<string, unknown>> };
      expect(input.pairs[0].mode).toBe("regex");
    } finally {
      vi.useRealTimers();
    }
  });

  test("keeps the source pair number when an earlier pair has no find string", async () => {
    mockOperationsAvailable();
    vi.spyOn(ipc, "opPreview").mockResolvedValue({
      text: "Hello there\nSecond line",
      counts: [1],
    });

    render(<CollectionView />);
    await openTool();
    vi.useFakeTimers();
    try {
      fireEvent.click(screen.getByRole("button", { name: "Add pair" }));
      const finds = screen.getAllByLabelText("Find");
      const replaces = screen.getAllByLabelText("Replace with");
      // The first pair stays empty; the second one is the visible row edited.
      fireEvent.change(finds[1], { target: { value: "world" } });
      fireEvent.change(replaces[1], { target: { value: "there" } });

      await act(async () => {
        await vi.advanceTimersByTimeAsync(275);
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      expect(screen.getByText("Pair 2: 1 replacements")).toBeInTheDocument();
      expect(screen.queryByText("Pair 1: 1 replacements")).toBeNull();
    } finally {
      vi.useRealTimers();
    }
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
    const cancelSpy = vi.spyOn(ipc, "opPreviewCancel").mockResolvedValue(undefined);
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
    // The dialog closes after a successful apply; nothing is left to cancel.
    expect(cancelSpy).not.toHaveBeenCalled();
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
    vi.stubGlobal("fetch", fetchSpy);

    render(<CollectionView />);
    await openTool();
    fillPair("world", "there");

    await waitForDiff("Hello there");
    expect(fetchSpy).toHaveBeenCalledWith(
      expect.stringMatching(/^reimagine-asset:\/\/localhost\/preview\/abc123\/out\.md\?v=\d+$/),
      { signal: expect.any(AbortSignal) },
    );
  });

  test("aborts a pending preview document fetch on close", async () => {
    mockOperationsAvailable();
    vi.spyOn(ipc, "opPreview").mockResolvedValue({
      doc: "reimagine-asset://preview/abc123/out.md",
      counts: [1],
    });
    let fetchSignal: AbortSignal | undefined;
    const fetchSpy = vi.fn((_url: string, init?: RequestInit) => {
      fetchSignal = init?.signal ?? undefined;
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          reject(new DOMException("Aborted", "AbortError"));
        });
      });
    });
    vi.stubGlobal("fetch", fetchSpy);

    render(<CollectionView />);
    await openTool();
    fillPair("world", "there");
    await waitFor(() => {
      expect(fetchSpy).toHaveBeenCalled();
    });

    fireEvent.click(screen.getByRole("button", { name: /^cancel$/i }));
    await waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });
    expect(fetchSignal?.aborted).toBe(true);
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

  test("cancels an in-flight preview on close and ignores its late response", async () => {
    mockOperationsAvailable();
    let resolvePreview: (value: unknown) => void = () => {};
    const previewSpy = vi.spyOn(ipc, "opPreview").mockImplementation(
      () =>
        new Promise((resolve) => {
          resolvePreview = resolve;
        }),
    );
    const cancelSpy = vi.spyOn(ipc, "opPreviewCancel").mockResolvedValue(undefined);
    const opInvokeSpy = mockCollection(() => ({
      id: "col_123",
      name: "Working draft",
      version: 2,
      slots: [{ id: "s1", asset_id: "ast_1" }],
    }));

    render(<CollectionView />);
    await openTool();
    fillPair("world", "there");
    await waitFor(() => {
      expect(previewSpy).toHaveBeenCalledTimes(1);
    });
    const previewKey = previewSpy.mock.calls[0][3];

    fireEvent.click(screen.getByRole("button", { name: /^cancel$/i }));
    await waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });
    expect(cancelSpy).toHaveBeenCalledTimes(1);
    expect(cancelSpy).toHaveBeenCalledWith(previewKey);

    await act(async () => {
      resolvePreview({ text: "Hello there\nSecond line", counts: [1] });
    });
    const names = opInvokeSpy.mock.calls.map(([, name]) => name);
    expect(names).not.toContain("text.replace");
    expect(names).not.toContain("collection.set_slot");

    // Reopening starts clean: the late response never became a diff.
    await openTool();
    expect(screen.queryByTestId("text-tool-diff")).toBeNull();
  });

  test("keeps a newer preview when an older edit resolves after it", async () => {
    mockOperationsAvailable();
    const resolvers: Array<(value: unknown) => void> = [];
    const previewSpy = vi.spyOn(ipc, "opPreview").mockImplementation(
      () =>
        new Promise((resolve) => {
          resolvers.push(resolve);
        }),
    );

    render(<CollectionView />);
    await openTool();

    fillPair("world", "first");
    await waitFor(() => {
      expect(previewSpy).toHaveBeenCalledTimes(1);
    });
    const previewKey = previewSpy.mock.calls[0][3];

    // The second edit restarts the debounce but keeps the dialog's key.
    fireEvent.change(screen.getByLabelText("Replace with"), { target: { value: "second" } });
    await waitFor(() => {
      expect(previewSpy).toHaveBeenCalledTimes(2);
    });
    expect(previewSpy.mock.calls[1][3]).toBe(previewKey);

    // The newer request settles first even though it was sent last.
    await act(async () => {
      resolvers[1]({ text: "Hello second\nSecond line", counts: [1] });
      await Promise.resolve();
    });
    await waitForDiff("Hello second");

    // The older request settles while the dialog is still open; it must not
    // overwrite the newer diff.
    await act(async () => {
      resolvers[0]({ text: "Hello first\nSecond line", counts: [1] });
      await Promise.resolve();
    });
    expect(screen.queryByText(/Hello first/)).toBeNull();
    expect(screen.getByTestId("text-tool-diff")).toHaveTextContent("Hello second");
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  test("does not dismiss on Escape while applying", async () => {
    mockOperationsAvailable();
    vi.spyOn(ipc, "opPreview").mockResolvedValue({
      text: "Hello there\nSecond line",
      counts: [1],
    });
    const cancelSpy = vi.spyOn(ipc, "opPreviewCancel").mockResolvedValue(undefined);
    let resolveReplace: (output: unknown) => void = () => {};
    const opInvokeSpy = mockCollection(
      () => ({
        id: "col_123",
        name: "Working draft",
        version: 2,
        slots: [{ id: "s1", asset_id: "ast_1" }],
      }),
      {
        "text.replace": () =>
          new Promise((resolve) => {
            resolveReplace = resolve;
          }),
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
      expect(opInvokeSpy.mock.calls.some(([, name]) => name === "text.replace")).toBe(true);
    });

    fireEvent.keyDown(document, { key: "Escape" });
    await act(async () => {
      await Promise.resolve();
    });
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    // An apply in progress owns the dialog, so Escape neither closes nor
    // cancels a preview.
    expect(cancelSpy).not.toHaveBeenCalled();

    resolveReplace({ doc: "ast_new", counts: [1] });
    await waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });
    expect(opInvokeSpy).toHaveBeenCalledWith("default", "collection.set_slot", {
      collection_id: "col_123",
      slot_id: "s1",
      asset: "ast_new",
    });
    expect(cancelSpy).not.toHaveBeenCalled();
  });

  test("hides the tool when the slot preview is not textual", async () => {
    mockOperationsAvailable();
    vi.spyOn(ipc, "readAssetPreview").mockResolvedValue({
      url: "asset-url",
      mediaType: "image/png",
      text: null,
    });

    render(<CollectionView />);
    await waitFor(() => {
      expect(screen.getByText("image/png")).toBeInTheDocument();
    });
    await settleOperations();
    expect(textToolButton()).toBeNull();
  });

  test("surfaces an op_list failure and retries on demand", async () => {
    const operations = await mockOpList({ projectId: "default" });
    const opListSpy = vi
      .spyOn(ipc, "opList")
      .mockRejectedValueOnce(new Error("host starting"))
      .mockRejectedValueOnce(new Error("still starting"))
      .mockResolvedValue(operations);

    render(<CollectionView />);
    await waitFor(
      () => {
        expect(screen.getByText("Could not load the text tools.")).toBeInTheDocument();
      },
      { timeout: 2000 },
    );
    expect(textToolButton()).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() => {
      expect(textToolButton()).toBeInTheDocument();
    });
    expect(opListSpy).toHaveBeenCalledTimes(3);
  });

  test("renders unknown schema fields and enum values verbatim", async () => {
    const operations = await mockOpList({ projectId: "default" });
    const customized = operations.map((operation) => {
      if (operation.spec.name !== "text.replace") return operation;
      const schema = JSON.parse(JSON.stringify(operation.spec.input_schema)) as {
        properties: { pairs: { items: { properties: Record<string, unknown> } } };
      };
      const properties = schema.properties.pairs.items.properties;
      properties.note_field = { type: "string" };
      (properties.mode as { enum: string[] }).enum = ["literal", "word", "regex", "custom_mode"];
      return { ...operation, spec: { ...operation.spec, input_schema: schema } };
    });
    vi.spyOn(ipc, "opList").mockResolvedValue(customized);

    render(<CollectionView />);
    await openTool();

    expect(screen.getByLabelText("note_field")).toBeInTheDocument();
    // An unknown enum value is shown as sent, not as the "Literal" fallback.
    expect(screen.getByRole("option", { name: "custom_mode" })).toBeInTheDocument();
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
