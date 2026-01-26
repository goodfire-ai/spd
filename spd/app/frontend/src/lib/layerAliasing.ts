/**
 * Layer aliasing system - transforms internal module names to human-readable aliases.
 *
 * Formats:
 * - Internal: "h.0.mlp.c_fc", "h.1.attn.q_proj"
 * - Aliased: "L0.mlp.in", "L1.attn.q"
 *
 * Handles multiple architectures:
 * - GPT-2: c_fc -> mlp.in, down_proj -> mlp.out
 * - Llama SwiGLU: gate_proj -> mlp.gate, up_proj -> mlp.up, down_proj -> mlp.down
 * - Attention: q_proj -> attn.q, k_proj -> attn.k, v_proj -> attn.v, o_proj -> attn.o
 * - Special: lm_head -> W_U, wte/output unchanged
 */

type Architecture = "gpt2" | "llama" | "unknown";

/** Mapping of internal module names to aliases by architecture */
const ALIASES: Record<Architecture, Record<string, string>> = {
    gpt2: {
        // MLP
        c_fc: "in",
        down_proj: "out",
        // Attention
        q_proj: "q",
        k_proj: "k",
        v_proj: "v",
        o_proj: "o",
    },
    llama: {
        // MLP (SwiGLU)
        gate_proj: "gate",
        up_proj: "up",
        down_proj: "down",
        // Attention
        q_proj: "q",
        k_proj: "k",
        v_proj: "v",
        o_proj: "o",
    },
    unknown: {
        // Fallback - just do attention mappings
        q_proj: "q",
        k_proj: "k",
        v_proj: "v",
        o_proj: "o",
    },
};

/** Special layers with fixed display names */
const SPECIAL_LAYERS: Record<string, string> = {
    lm_head: "W_U",
    wte: "wte",
    output: "output",
};

/**
 * Detect architecture from layer name.
 * Llama has gate_proj/up_proj, GPT-2 has c_fc.
 */
function detectArchitecture(layer: string): Architecture {
    if (layer.includes("gate_proj") || layer.includes("up_proj")) {
        return "llama";
    }
    if (layer.includes("c_fc")) {
        return "gpt2";
    }
    return "unknown";
}

/**
 * Parse a layer name into components.
 * Returns null for special layers (wte, output, lm_head) or unrecognized formats.
 */
function parseLayerName(layer: string): { block: number; moduleType: string; submodule: string } | null {
    if (layer in SPECIAL_LAYERS) {
        return null;
    }

    const match = layer.match(/^h\.(\d+)\.(attn|mlp)\.(\w+)$/);
    if (!match) {
        return null;
    }

    const [, blockStr, moduleType, submodule] = match;
    return {
        block: parseInt(blockStr),
        moduleType,
        submodule,
    };
}

/**
 * Transform a layer name to its aliased form.
 *
 * Examples:
 * - "h.0.mlp.c_fc" -> "L0.mlp.in"
 * - "h.2.attn.q_proj" -> "L2.attn.q"
 * - "lm_head" -> "W_U"
 * - "wte" -> "wte"
 */
export function getLayerAlias(layer: string): string {
    if (layer in SPECIAL_LAYERS) {
        return SPECIAL_LAYERS[layer];
    }

    const parsed = parseLayerName(layer);
    if (!parsed) {
        return layer;
    }

    const arch = detectArchitecture(layer);
    const alias = ALIASES[arch][parsed.submodule];

    if (!alias) {
        return `L${parsed.block}.${parsed.moduleType}.${parsed.submodule}`;
    }

    return `L${parsed.block}.${parsed.moduleType}.${alias}`;
}

/**
 * Get a short row label for grouped display in graphs.
 * Used for Y-axis labels where space is limited.
 *
 * Examples:
 * - "h.0.mlp.c_fc" -> "L0.in"
 * - "h.2.attn.q_proj" -> "L2.q" (but QKV are typically grouped as "L2.qkv")
 * - "lm_head" -> "W_U"
 */
export function getRowLabel(layer: string, isQkvGroup = false): string {
    if (layer in SPECIAL_LAYERS) {
        return SPECIAL_LAYERS[layer];
    }

    const parsed = parseLayerName(layer);
    if (!parsed) {
        return layer;
    }

    if (isQkvGroup) {
        return `L${parsed.block}.qkv`;
    }

    const arch = detectArchitecture(layer);
    const alias = ALIASES[arch][parsed.submodule];

    if (!alias) {
        return `L${parsed.block}.${parsed.submodule}`;
    }

    return `L${parsed.block}.${alias}`;
}

/**
 * Format a node key with aliased layer names.
 *
 * Node keys are "layer:seq:cIdx" or "layer:cIdx" format.
 *
 * Examples:
 * - "h.0.mlp.c_fc:3:5" -> "L0.mlp.in:3:5"
 * - "h.1.attn.q_proj:2:10" -> "L1.attn.q:2:10"
 */
export function formatNodeKeyWithAliases(nodeKey: string): string {
    const parts = nodeKey.split(":");
    const layer = parts[0];
    const aliasedLayer = getLayerAlias(layer);
    return [aliasedLayer, ...parts.slice(1)].join(":");
}
