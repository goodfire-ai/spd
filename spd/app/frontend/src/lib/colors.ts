/**
 * Centralized color definitions for graph visualization.
 * These match the CSS variables in app.css but are available for inline styles in SVG elements.
 */

export const colors = {
    // Text - punchy contrast
    textPrimary: "#111111",
    textSecondary: "#555555",
    textMuted: "#999999",

    // Status colors for edges/data - vivid
    positive: "#2563eb",
    negative: "#dc2626",

    // Output node gradient (green) - vivid green
    outputBase: { r: 22, g: 163, b: 74 },

    // Token highlight - vivid green
    tokenHighlight: { r: 22, g: 163, b: 74 },
    tokenHighlightOpacity: 0.4,

    // Node default
    nodeDefault: "#6b7280",

    // Accent (for active states) - blue
    accent: "#2563eb",

    // Set overlap visualization (A/B/intersection)
    setOverlap: {
        self: "rgb(20, 184, 166)", // teal - A-only
        both: "rgb(100, 116, 139)", // slate - intersection
        other: "rgb(249, 115, 22)", // orange - B-only
    },
} as const;

/** Get output node fill color based on probability */
export function getOutputNodeColor(prob: number): string {
    const { r, g, b } = colors.outputBase;
    return `rgb(${r + Math.round(prob * 10)}, ${g + Math.round(prob * 25)}, ${b + Math.round(prob * 16)})`;
}

/** Get edge color based on value sign */
export function getEdgeColor(val: number): string {
    return val > 0 ? colors.positive : colors.negative;
}

/** Get node color for subcomponent activation (blue=positive, red=negative) */
export function getSubcompActColor(val: number): string {
    return val >= 0 ? colors.positive : colors.negative;
}

/** Get token highlight background for CI values (0-1, green) */
export function getTokenHighlightBg(ci: number): string {
    const { r, g, b } = colors.tokenHighlight;
    return `rgba(${r},${g},${b},${ci * colors.tokenHighlightOpacity})`;
}

/** Get color for component activations (blue for positive, red for negative) */
export function getComponentActivationColor(value: number, normalizedAbs: number): string {
    if (value >= 0) {
        return `rgba(37, 99, 235, ${normalizedAbs})`; // blue
    } else {
        return `rgba(220, 38, 38, ${normalizedAbs})`; // red
    }
}

/** Compute the max absolute value across all component activations (for normalization) */
export function computeMaxAbsComponentAct(exampleComponentActs: number[][]): number {
    let max = 0;
    for (const row of exampleComponentActs) {
        for (const val of row) {
            const abs = Math.abs(val);
            if (abs > max) max = abs;
        }
    }
    return max === 0 ? 1 : max;
}

/** Get output header gradient background based on probability */
export function getOutputHeaderColor(prob: number): string {
    const { r, g, b } = colors.outputBase;
    const opacity = Math.min(0.8, prob + 0.05);
    return `rgba(${r},${g},${b},${opacity})`;
}

/** Background color with opacity for overlays */
export const bgBaseRgb = { r: 255, g: 255, b: 255 };
