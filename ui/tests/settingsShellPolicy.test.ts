import { expect, test } from "bun:test";

import {
  shouldForceRuntimeCollapsed,
  shouldSuppressContextPanels,
} from "../src/components/layout/settingsShellPolicy";

test("settings mode suppresses lower-priority shell context", () => {
  expect(shouldSuppressContextPanels("Settings")).toBe(true);
  expect(shouldForceRuntimeCollapsed("Settings")).toBe(true);
});

test("ordinary explorer panels keep contextual chrome available", () => {
  expect(shouldSuppressContextPanels("Graph")).toBe(false);
  expect(shouldForceRuntimeCollapsed("Graph")).toBe(false);
  expect(shouldSuppressContextPanels(null)).toBe(false);
  expect(shouldForceRuntimeCollapsed(null)).toBe(false);
});
