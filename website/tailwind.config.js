/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { 950: "#080b10", 900: "#0a0d13", 800: "#0f141d", 700: "#161d29", 600: "#1c2533" },
        accent: { sky: "#38bdf8", indigo: "#818cf8", violet: "#c084fc", pink: "#f0abfc" },
      },
      fontFamily: {
        sans: ["Inter", "Segoe UI", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      keyframes: {
        float: { "0%,100%": { transform: "translateY(0)" }, "50%": { transform: "translateY(-8px)" } },
        fadeup: { "0%": { opacity: 0, transform: "translateY(14px)" }, "100%": { opacity: 1, transform: "translateY(0)" } },
        blink: { "0%,100%": { opacity: 1 }, "50%": { opacity: 0 } },
        "gradient-pan": { "0%,100%": { backgroundPosition: "0% 50%" }, "50%": { backgroundPosition: "100% 50%" } },
        "spin-slow": { to: { transform: "rotate(360deg)" } },
        aurora: {
          "0%": { transform: "translate(0,0) scale(1)" },
          "33%": { transform: "translate(3%,-4%) scale(1.08)" },
          "66%": { transform: "translate(-3%,3%) scale(0.96)" },
          "100%": { transform: "translate(0,0) scale(1)" },
        },
        shimmer: { "100%": { transform: "translateX(100%)" } },
      },
      animation: {
        float: "float 7s ease-in-out infinite",
        fadeup: "fadeup 0.7s ease-out both",
        blink: "blink 1s step-end infinite",
        "gradient-pan": "gradient-pan 6s ease infinite",
        "spin-slow": "spin-slow 18s linear infinite",
        aurora: "aurora 18s ease-in-out infinite",
        shimmer: "shimmer 2.5s infinite",
      },
    },
  },
  plugins: [],
};
