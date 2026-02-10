/**
 * Graph layout utilities for canonical transformer addresses.
 *
 * Canonical address format:
 *   "wte"              — embedding
 *   "output"           — unembed / logits
 *   "{block}.{sublayer}.{projection}" — e.g. "0.attn.q", "2.mlp.down"
 *
 * Node key format:
 *   "{layer}:{seqIdx}:{cIdx}" — e.g. "0.attn.q:3:5", "wte:0:0"
 */

export type LayerInfo = {
    name: string;
    block: number; // -1 for wte, Infinity for output
    sublayer: string; // "attn" | "attn_fused" | "mlp" | "glu" | "wte" | "output"
    projection: string | null; // "q" | "k" | "v" | "o" | "qkv" | "up" | "down" | "gate" | null
};

const SUBLAYER_ORDER = ["attn", "attn_fused", "glu", "mlp"];

// Projections that share a row and get grouped horizontally
const GROUPED_PROJECTIONS: Record<string, string[]> = {
    attn: ["q", "k", "v"],
    glu: ["gate", "up"],
};

export function parseLayer(name: string): LayerInfo {
    if (name === "wte") return { name, block: -1, sublayer: "wte", projection: null };
    if (name === "output") return { name, block: Infinity, sublayer: "output", projection: null };

    const parts = name.split(".");
    return {
        name,
        block: +parts[0],
        sublayer: parts[1],
        projection: parts[2],
    };
}

/**
 * Row key: layers that share the same visual row.
 * q/k/v share "0.attn", gate/up share "0.glu", etc.
 */
export function getRowKey(layer: string): string {
    const info = parseLayer(layer);
    if (info.sublayer === "wte" || info.sublayer === "output") return layer;
    return `${info.block}.${info.sublayer}`;
}

/**
 * Row label for display.
 */
export function getRowLabel(layer: string): string {
    if (layer === "wte") return "wte";
    if (layer === "output") return "output";

    const info = parseLayer(layer);
    return `${info.block}.${info.sublayer}`;
}

/**
 * Sort row keys: wte at bottom, output at top, blocks in between.
 * Within a block, sublayers follow SUBLAYER_ORDER.
 */
export function sortRows(rows: string[]): string[] {
    // TODO(oli) adjust me for canonical addresses
    return [...rows].sort((a, b) => {
        // Parse row keys (which are "block.sublayer" format)
        const blockA = a === "wte" ? -1 : a === "output" ? Infinity : +a.split(".")[0];
        const blockB = b === "wte" ? -1 : b === "output" ? Infinity : +b.split(".")[0];

        if (blockA !== blockB) return blockA - blockB;

        const sublayerA = a.split(".")[1] ?? "";
        const sublayerB = b.split(".")[1] ?? "";
        return SUBLAYER_ORDER.indexOf(sublayerA) - SUBLAYER_ORDER.indexOf(sublayerB);
    });
}

/**
 * Get the grouped projections for a sublayer, if any.
 * Returns null if no grouping (each projection gets its own horizontal space).
 */
export function getGroupProjections(sublayer: string): string[] | null {
    return GROUPED_PROJECTIONS[sublayer] ?? null;
}

/**
 * Build the full layer address from block + sublayer + projection.
 */
export function buildLayerAddress(block: number, sublayer: string, projection: string): string {
    return `${block}.${sublayer}.${projection}`;
}
