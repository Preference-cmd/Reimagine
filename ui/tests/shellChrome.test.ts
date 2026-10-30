import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import {
  applyDocumentLang,
  formatRelativeTime,
  showsSessionPanel,
} from "../src/components/layout/shellChrome";

describe("showsSessionPanel", () => {
  test("shows the panel on canvas and library modes only", () => {
    expect(showsSessionPanel("/")).toBe(true);
    expect(showsSessionPanel("/library")).toBe(true);
    expect(showsSessionPanel("/models")).toBe(false);
    expect(showsSessionPanel("/settings")).toBe(false);
    expect(showsSessionPanel("/collections/col_1")).toBe(false);
    expect(showsSessionPanel("/library/extra")).toBe(false);
  });
});

describe("applyDocumentLang", () => {
  test("maps the UI locale onto the document language", () => {
    applyDocumentLang("zh");
    expect(document.documentElement.lang).toBe("zh-CN");
    applyDocumentLang("en");
    expect(document.documentElement.lang).toBe("en");
  });
});

describe("formatRelativeTime", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-10-29T12:00:00Z"));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  test("formats elapsed time in the requested locale", () => {
    expect(formatRelativeTime("2026-10-26T12:00:00Z", "en")).toBe("3d ago");
    expect(formatRelativeTime("2026-10-26T12:00:00Z", "zh-CN")).toBe("3天前");
  });

  test("formats recent and future timestamps", () => {
    expect(formatRelativeTime("2026-10-29T12:00:00Z", "en")).toBe("now");
    expect(formatRelativeTime("2026-11-01T12:00:00Z", "en")).toBe("in 3d");
  });

  test("returns an empty label for an unparseable timestamp", () => {
    expect(formatRelativeTime("not-a-date", "en")).toBe("");
  });
});
