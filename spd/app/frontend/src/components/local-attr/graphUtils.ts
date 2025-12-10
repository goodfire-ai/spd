/**
 * Utility functions for the LocalAttributionsGraph component.
 */

/** Linear interpolation between min and max. */
export function lerp(min: number, max: number, t: number): number {
    return min + (max - min) * t;
}

/** Hash a string to a number (for seeded random). */
export function hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = (hash << 5) - hash + char;
        hash = hash & hash;
    }
    return Math.abs(hash);
}

/** Shuffle an array in place using a seeded random number generator. */
export function seededShuffle<T>(arr: T[], seed: number): T[] {
    const random = () => {
        seed |= 0;
        seed = (seed + 0x6d2b79f5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };

    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
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
