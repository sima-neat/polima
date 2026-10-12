import { defineConfig } from "vite";
import preact from "@preact/preset-vite";
import { resolve } from "node:path";

export default defineConfig({
  plugins: [preact()],
  build: {
    outDir: resolve(import.meta.dirname, "../static"),
    emptyOutDir: true,
  },
});
