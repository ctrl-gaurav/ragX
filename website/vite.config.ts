import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages project site: https://ctrl-gaurav.github.io/ragX/ (SITE_BASE=/ when served from a domain root)
export default defineConfig({
  plugins: [react()],
  base: process.env.SITE_BASE || "/ragX/",
  build: { outDir: "dist" },
});
