/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#EEF3F1",
        surface: "#FFFFFF",
        panel: "#F4F8F6",
        line: "#E2EAE7",
        ink: {
          DEFAULT: "#0D1B20",
          soft: "#54666D",
          faint: "#8CA0A5",
        },
        primary: {
          DEFAULT: "#0E9F8E",
          50: "#E6F6F3",
          100: "#C6EBE5",
          200: "#93DBD0",
          300: "#54C5B6",
          400: "#22AE9D",
          500: "#0E9F8E",
          600: "#0B8273",
          700: "#0B6458",
          800: "#0B4D45",
          900: "#0A3A34",
        },
        accent: "#12B5C4",
        gold: "#B08A45",
        sev: {
          low: "#1A9E54",
          med: "#E0A008",
          high: "#F07316",
          crit: "#DC2A28",
        },
        info: "#2F6FED",
      },
      fontFamily: {
        sans: ['"IBM Plex Sans Arabic"', '"IBM Plex Sans"', "system-ui", "sans-serif"],
        mono: ['"IBM Plex Mono"', "ui-monospace", "monospace"],
      },
      boxShadow: {
        soft: "0 1px 2px rgba(13,27,32,.04), 0 10px 30px -16px rgba(13,27,32,.18)",
        card: "0 1px 0 rgba(13,27,32,.03), 0 4px 18px -10px rgba(13,27,32,.16)",
        lift: "0 18px 40px -18px rgba(11,100,88,.40)",
        ring: "0 0 0 4px rgba(14,159,142,.12)",
      },
      borderRadius: {
        xl: "14px",
        "2xl": "20px",
        "3xl": "26px",
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "scan": {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(220%)" },
        },
        "pulse-ring": {
          "0%": { boxShadow: "0 0 0 0 rgba(14,159,142,.45)" },
          "70%": { boxShadow: "0 0 0 12px rgba(14,159,142,0)" },
          "100%": { boxShadow: "0 0 0 0 rgba(14,159,142,0)" },
        },
        "ticker": {
          "0%": { transform: "translateX(0)" },
          "100%": { transform: "translateX(-50%)" },
        },
      },
      animation: {
        "fade-up": "fade-up .5s cubic-bezier(.22,.61,.36,1) both",
        scan: "scan 2.6s ease-in-out infinite",
        "pulse-ring": "pulse-ring 2s ease-out infinite",
        ticker: "ticker 30s linear infinite",
      },
    },
  },
  plugins: [],
};
