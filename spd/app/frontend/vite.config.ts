import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

const backendUrl = process.env.VITE_API_URL;
if (!backendUrl) {
    throw new Error("VITE_API_URL environment variable is required. Run the app via `make app` or `python -m spd.app.run_app`.");
}

// https://vite.dev/config/
export default defineConfig({
    plugins: [svelte()],
    server: {
        proxy: {
            "/api": {
                target: backendUrl,
                changeOrigin: true,
            },
        },
    },
});
