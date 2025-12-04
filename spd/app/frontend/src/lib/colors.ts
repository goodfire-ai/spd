/**
 * Centralized color definitions for graph visualization.
 * These match the CSS variables in app.css but are available for inline styles in SVG elements.
 */

export const colors = {
    // Text
    textPrimary: "#1a1918",
    textSecondary: "#4a4844",
    textMuted: "#8a867e",

    // Status colors for edges/data
    positive: "#2868a0",
    negative: "#c43c3c",

    // Output node gradient (green) - brighter for better visibility
    outputBase: { r: 34, g: 170, b: 90 },

    // Token highlight - brighter green
    tokenHighlight: { r: 34, g: 170, b: 90 },
    tokenHighlightOpacity: 0.4,

    // Node default
    nodeDefault: "#8a867e",

    // Accent (for active states)
    accent: "#c45a28",
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

/** Get token highlight background */
export function getTokenHighlightBg(ci: number): string {
    const { r, g, b } = colors.tokenHighlight;
    return `rgba(${r},${g},${b},${ci * colors.tokenHighlightOpacity})`;
}

/** Get output header gradient background based on probability */
export function getOutputHeaderGradient(prob: number): string {
    const { r, g, b } = colors.outputBase;
    const opacity = Math.min(0.8, prob + 0.1);
    return `linear-gradient(90deg, rgba(${r},${g},${b},${opacity}) 0%, rgba(${r},${g},${b},0.1) 100%)`;
}

/** Background color with opacity for overlays */
export const bgBaseRgb = { r: 245, g: 243, b: 239 };
