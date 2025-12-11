/**
 * Utility functions for the LocalAttributionsGraph component.
 */

/** Linear interpolation between min and max. */
export function lerp(min: number, max: number, t: number): number {
    return min + (max - min) * t;
}

/** Calculate tooltip position that stays within viewport bounds. */
export function calcTooltipPos(mouseX: number, mouseY: number): { x: number; y: number } {
    const padding = 15;
    // Tooltip has max-height: 80vh, so use that as the estimate
    const tooltipHeight = typeof window !== "undefined" ? window.innerHeight * 0.8 : 400;

    let left = mouseX + padding;
    let top = mouseY + padding;

    if (typeof window !== "undefined") {
        // If mouse is in right half of screen, position tooltip to the left
        if (mouseX > window.innerWidth / 2) {
            left = padding;
        }
        // Clamp to bottom of screen (don't flip, just constrain)
        if (top + tooltipHeight > window.innerHeight) {
            top = window.innerHeight - tooltipHeight - padding;
        }
    }
    return { x: Math.max(0, left), y: Math.max(0, top) };
}
