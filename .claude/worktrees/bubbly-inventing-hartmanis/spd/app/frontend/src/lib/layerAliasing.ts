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
 * - Special: lm_head -> W_U, embed/output unchanged
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
    embed: "embed",
    output: "output",
};

// Cache for detected architecture from the full model
let cachedArchitecture: Architecture | null = null;

/**
 * Detect architecture from a collection of layer names.
 * Llama has gate_proj/up_proj, GPT-2 has c_fc.
 *
 * This should be called once with all available layer names to establish
 * the architecture for the session, ensuring down_proj is aliased correctly.
 */
export function detectArchitectureFromLayers(layers: string[]): Architecture {
    const hasLlamaLayers = layers.some((layer) => layer.includes("gate_proj") || layer.includes("up_proj"));
    if (hasLlamaLayers) {
        return "llama";
    }

    const hasGPT2Layers = layers.some((layer) => layer.includes("c_fc"));
    if (hasGPT2Layers) {
        return "gpt2";
    }

    return "unknown";
}

/**
 * Set the architecture for aliasing operations.
 * Call this when you have access to all layer names (e.g., when loading a graph).
 */
export function setArchitecture(layers: string[]): void {
    cachedArchitecture = detectArchitectureFromLayers(layers);
}

/**
 * Detect architecture from layer name.
 * Uses cached architecture if available (set via setArchitecture()),
 * otherwise falls back to single-layer detection.
 *
 * Note: down_proj appears in both architectures with different meanings:
 * - GPT-2: down_proj -> "out" (second MLP projection)
 * - Llama: down_proj -> "down" (third MLP projection after gate/up)
 *
 * Single-layer detection cannot distinguish these cases reliably.
 */
function detectArchitecture(layer: string): Architecture {
    // Use cached architecture if available
    if (cachedArchitecture !== null) {
        return cachedArchitecture;
    }

    // Fallback: single-layer detection (less reliable for down_proj)
    if (layer.includes("gate_proj") || layer.includes("up_proj")) {
        return "llama";
    }
    if (layer.includes("c_fc")) {
        return "gpt2";
    }
    // down_proj is ambiguous without context, default to GPT-2
    if (layer.includes("down_proj")) {
        return "gpt2";
    }
    return "unknown";
}

/**
 * Parse a layer name into components.
 * Returns null for special layers (embed, output, lm_head) or unrecognized formats.
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
 * - "embed" -> "embed"
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
 * Get a row label for grouped display in graphs.
 *
 * @param layer - Internal layer name (e.g., "h.0.mlp.c_fc")
 * @param isQkvGroup - Whether this represents a grouped QKV row
 * @returns Label (e.g., "L0.mlp.in", "L2.attn.qkv")
 *
 * @example
 * getAliasedRowLabel("h.0.mlp.c_fc") // => "L0.mlp.in"
 * getAliasedRowLabel("h.2.attn.q_proj", true) // => "L2.attn.qkv"
 */
export function getAliasedRowLabel(layer: string, isQkvGroup = false): string {
    if (layer in SPECIAL_LAYERS) {
        return SPECIAL_LAYERS[layer];
    }

    const parsed = parseLayerName(layer);
    if (!parsed) {
        return layer;
    }

    if (isQkvGroup) {
        return `L${parsed.block}.${parsed.moduleType}.qkv`;
    }

    const arch = detectArchitecture(layer);
    const alias = ALIASES[arch][parsed.submodule];

    if (!alias) {
        return `L${parsed.block}.${parsed.moduleType}.${parsed.submodule}`;
    }

    return `L${parsed.block}.${parsed.moduleType}.${alias}`;
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
