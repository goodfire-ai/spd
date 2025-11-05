// Temporary type augmentation for Svelte 5 onclick events
// Remove this file once Svelte type definitions are updated

declare module 'svelte/elements' {
    interface HTMLButtonAttributes {
        onclick?: ((event: MouseEvent) => void) | undefined;
    }
}
