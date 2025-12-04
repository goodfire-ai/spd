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
    nodeDefault: "#888888",

    // Accent (for active states) - blue
    accent: "#2563eb",
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
export const bgBaseRgb = { r: 255, g: 255, b: 255 };
