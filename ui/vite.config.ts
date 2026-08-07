/// <reference types="vitest" />
import { defineConfig } from "vite";
import path from "node:path";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { paraglideVitePlugin } from "@inlang/paraglide-js";
import { visualizer } from "rollup-plugin-visualizer";

// @ts-expect-error process is a nodejs global
const host = process.env.TAURI_DEV_HOST;
const analyze = process.env.ANALYZE === "true";

// https://vite.dev/config/
export default defineConfig(async () => ({
  plugins: [
    react(),
    tailwindcss(),
    paraglideVitePlugin({
      project: "./project.inlang",
      outdir: "./src/paraglide",
      emitTsDeclarations: true,
    }),
    ...(analyze
      ? [
          visualizer({
            open: true,
            filename: "dist/stats.html",
            gzipSize: true,
          }),
        ]
      : []),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "$paraglide/messages": path.resolve(__dirname, "./src/paraglide/messages.js"),
      "$paraglide/runtime": path.resolve(__dirname, "./src/paraglide/runtime.js"),
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          "vendor-react": ["react", "react-dom"],
          "vendor-xyflow": ["@xyflow/react"],
        },
      },
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/**/*.test.{ts,tsx}"],
  },
  // Vite options tailored for Tauri development and only applied in `tauri dev` or `tauri build`
  //
  // 1. prevent Vite from obscuring rust errors
  clearScreen: false,
  // 2. tauri expects a fixed port, fail if that port is not available
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: 1421,
        }
      : undefined,
    watch: {
      // 3. tell Vite to ignore watching `src-tauri`
      ignored: ["**/src-tauri/**"],
    },
  },
}));
