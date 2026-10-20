import { defineConfig } from "vitepress";

// GitHub Pages serves the docs at https://ctrl-gaurav.github.io/ragX/docs/; a host serving the
// site from a domain root sets DOCS_BASE=/docs/ (and Netlify sets URL for the link home).
const BASE = process.env.DOCS_BASE || "/ragX/docs/";
const HOME = process.env.SITE_HOME || (process.env.URL ? `${process.env.URL}/` : "https://ctrl-gaurav.github.io/ragX/");

export default defineConfig({
  title: "ragX",
  titleTemplate: ":title · ragX docs",
  description: "ragX documentation: agentic retrieval with exact citations, from pip install to every option.",
  base: BASE,
  cleanUrls: true,
  lastUpdated: false,
  head: [
    ["link", { rel: "icon", type: "image/svg+xml", href: `${BASE}favicon.svg` }],
    ["meta", { name: "theme-color", content: "#080b10" }],
    ["meta", { property: "og:title", content: "ragX documentation" }],
    ["meta", { property: "og:description", content: "Agentic retrieval at scale, with exact citations." }],
  ],
  markdown: {
    theme: { light: "github-light", dark: "github-dark" },
    lineNumbers: false,
  },
  themeConfig: {
    logo: { light: "/logo-light.svg", dark: "/logo.svg" },
    logoLink: HOME, // the logo goes to the site's home page, not the docs home
    siteTitle: false,
    nav: [
      { text: "Guide", link: "/guide/quickstart" },
      { text: "Reference", link: "/reference/python" },
      { text: "Changelog", link: "/changelog" },
      { text: "Home ↗", link: HOME },
    ],
    sidebar: [
      {
        text: "Introduction",
        items: [
          { text: "What is ragX?", link: "/" },
          { text: "Installation", link: "/guide/installation" },
          { text: "Quickstart", link: "/guide/quickstart" },
        ],
      },
      {
        text: "Concepts",
        items: [
          { text: "How ragX works", link: "/guide/how-it-works" },
          { text: "Variants", link: "/guide/variants" },
          { text: "Retrievers", link: "/guide/retrievers" },
          { text: "Corpora and passages", link: "/guide/corpora" },
          { text: "Citations and grounding", link: "/guide/citations" },
        ],
      },
      {
        text: "Using ragX",
        items: [
          { text: "Models and providers", link: "/guide/models" },
          { text: "Command line", link: "/guide/cli" },
          { text: "OpenAI-compatible server", link: "/guide/server" },
          { text: "Streaming the trace", link: "/guide/streaming" },
          { text: "Security", link: "/guide/security" },
          { text: "Cost and speed", link: "/guide/performance" },
          { text: "The classic agent", link: "/guide/classic" },
        ],
      },
      {
        text: "Reference",
        items: [
          { text: "Python API", link: "/reference/python" },
          { text: "Configuration", link: "/reference/configuration" },
          { text: "Environment variables", link: "/reference/environment" },
        ],
      },
      {
        text: "Project",
        items: [
          { text: "FAQ", link: "/faq" },
          { text: "Changelog", link: "/changelog" },
          { text: "Citing ragX", link: "/citing" },
        ],
      },
    ],
    outline: { level: [2, 3], label: "On this page" },
    socialLinks: [{ icon: "github", link: "https://github.com/ctrl-gaurav/ragX" }],
    search: { provider: "local" },
    editLink: {
      pattern: "https://github.com/ctrl-gaurav/ragX/edit/main/docs/:path",
      text: "Edit this page on GitHub",
    },
    footer: { message: "Released under the Apache 2.0 License.", copyright: "© 2026 Gaurav Srivastava" },
  },
});
