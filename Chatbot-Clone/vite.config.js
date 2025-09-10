import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    lib: {
      entry: "src/main.jsx",          // ✅ correct entry
      name: "MyChatbot",              // ✅ attaches to window.MyChatbot
      fileName: "chatbot",            // will produce chatbot.iife.js
      formats: ["iife"],              // ✅ only IIFE build
    },
    rollupOptions: {
      // 👉 If you want React bundled inside, leave external empty
      // If you want React loaded separately via CDN, add react + react-dom here
      external: [],
      output: {
        globals: {
          react: "React",
          "react-dom/client": "ReactDOM",
        },
      },
    },
  },
  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
  },
});
