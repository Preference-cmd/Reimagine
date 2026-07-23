import { expect, test } from "bun:test";

import {
  EXPLORER_PANELS,
  getOverlayPolicy,
  isExplorerPanel,
  isSettingsPanel,
} from "../src/components/layout/overlayLayout";

test("recognizes explorer panels as a shared overlay group", () => {
  expect(EXPLORER_PANELS).toEqual(["Graph", "Models", "Runs", "Assets"]);
  expect(isExplorerPanel("Graph")).toBe(true);
  expect(isExplorerPanel("Settings")).toBe(false);
  expect(isExplorerPanel(null)).toBe(false);
});

test("recognizes settings as the modal overlay owner", () => {
  expect(isSettingsPanel("Settings")).toBe(true);
  expect(isSettingsPanel("Graph")).toBe(false);
  expect(isSettingsPanel(null)).toBe(false);
});

test("settings suppresses contextual overlays and collapses runtime details", () => {
  expect(getOverlayPolicy("Settings")).toEqual({
    explorerOpen: false,
    explorerView: null,
    forceRuntimeCollapsed: true,
    settingsOpen: true,
    suppressContextPanels: true,
  });
});

test("explorer panels coexist with contextual overlays", () => {
  expect(getOverlayPolicy("Models")).toEqual({
    explorerOpen: true,
    explorerView: "Models",
    forceRuntimeCollapsed: false,
    settingsOpen: false,
    suppressContextPanels: false,
  });
});

test("empty overlay state keeps chrome quiet", () => {
  expect(getOverlayPolicy(null)).toEqual({
    explorerOpen: false,
    explorerView: null,
    forceRuntimeCollapsed: false,
    settingsOpen: false,
    suppressContextPanels: false,
  });
});
