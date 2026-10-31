import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

const invokeMock = vi.fn();

vi.mock("@tauri-apps/api/core", () => ({
  invoke: (...args: unknown[]) => invokeMock(...args),
  convertFileSrc: (path: string, protocol: string) => `${protocol}://localhost/${path}`,
  Channel: class {},
}));

import { opList, opPreview, opPreviewCancel } from "../src/ipc/commands";
import { mockOpPreviewCancel } from "../src/ipc/mock";

describe("operation IPC wrappers", () => {
  beforeEach(() => {
    invokeMock.mockReset();
    invokeMock.mockResolvedValue(undefined);
    (window as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__ = {};
  });

  afterEach(() => {
    delete (window as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;
  });

  test("op_list and op_preview send the host command names and arguments", async () => {
    await opList("proj-1");
    expect(invokeMock).toHaveBeenNthCalledWith(1, "op_list", { projectId: "proj-1" });

    const input = { doc: "ast_1", pairs: [{ find: "a", replace: "b", mode: "literal" }] };
    await opPreview("proj-1", "text.replace", input, "pk_1");
    expect(invokeMock).toHaveBeenNthCalledWith(2, "op_preview", {
      projectId: "proj-1",
      name: "text.replace",
      input,
      previewKey: "pk_1",
    });
  });

  test("op_preview_cancel sends the preview key", async () => {
    await opPreviewCancel("pk_1");
    expect(invokeMock).toHaveBeenCalledWith("op_preview_cancel", { previewKey: "pk_1" });
  });

  test("the preview-cancel mock is a no-op", async () => {
    await expect(mockOpPreviewCancel({ previewKey: "pk_1" })).resolves.toBeUndefined();
  });
});
