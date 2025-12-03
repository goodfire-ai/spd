import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

/** @type {import("@sveltejs/vite-plugin-svelte").SvelteConfig} */
export default {
    // Consult https://svelte.dev/docs#compile-time-svelte-preprocess
    // for more information about preprocessors
    preprocess: vitePreprocess(),
    onwarn: (warning, handler) => {
        // Ignore a11y warnings for mouse events on SVG visualizations
        if (warning.code === "a11y_mouse_events_have_key_events") return;
        handler(warning);
    },
};
