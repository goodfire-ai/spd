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

/**
 * Sort component indices by importance (CI for internal nodes, probability for output nodes).
 * Returns a new sorted array (highest importance first).
 */
export function sortComponentsByImportance(
    components: number[],
    layer: string,
    seqIdx: number,
    nodeCiVals: Record<string, number>,
    outputProbs: Record<string, { prob: number }>,
): number[] {
    const isOutput = layer === "output";
    return [...components].sort((a, b) => {
        if (isOutput) {
            const keyA = `${seqIdx}:${a}`;
            const keyB = `${seqIdx}:${b}`;
            return (outputProbs[keyB]?.prob ?? 0) - (outputProbs[keyA]?.prob ?? 0);
        }
        const keyA = `${layer}:${seqIdx}:${a}`;
        const keyB = `${layer}:${seqIdx}:${b}`;
        return (nodeCiVals[keyB] ?? 0) - (nodeCiVals[keyA] ?? 0);
    });
}

/**
 * Compute X offsets for components given their display order.
 * Returns a map from component index to its X offset in pixels.
 */
export function computeComponentOffsets(
    sortedComponents: number[],
    componentSize: number,
    componentGap: number,
): Record<number, number> {
    const offsets: Record<number, number> = {};
    for (let i = 0; i < sortedComponents.length; i++) {
        offsets[sortedComponents[i]] = i * (componentSize + componentGap);
    }
    return offsets;
}
