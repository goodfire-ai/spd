/** Types for the prompt attributions visualizer */

// Server API types

export type PromptPreview = {
    id: number;
    token_ids: number[];
    tokens: string[];
    preview: string;
};

export type Edge = {
    src: string; // "layer:seq:cIdx"
    tgt: string; // "layer:seq:cIdx"
    val: number;
};

export type EdgeAttribution = {
    key: string; // "layer:seq:cIdx" for prompt or "layer:cIdx" for dataset
    value: number; // raw attribution value (positive or negative)
    normalizedMagnitude: number; // |value| / maxAbsValue, for color intensity (0-1)
};

export type OutputProbEntry = {
    prob: number; // CI-masked (SPD model) probability
    logit: number; // CI-masked (SPD model) raw logit
    target_prob: number; // Target model probability
    target_logit: number; // Target model raw logit
    token: string;
};

export type GraphType = "standard" | "optimized" | "manual";

export type GraphData = {
    id: number;
    graphType: GraphType;
    tokens: string[];
    edges: Edge[];
    edgesBySource: Map<string, Edge[]>; // nodeKey -> edges where this node is source
    edgesByTarget: Map<string, Edge[]>; // nodeKey -> edges where this node is target
    outputProbs: Record<string, OutputProbEntry>; // key is "seq:cIdx"
    nodeCiVals: Record<string, number>; // node key -> CI value (or output prob for output nodes or 1 for wte node)
    nodeSubcompActs: Record<string, number>; // node key -> subcomponent activation (v_i^T @ a)
    maxAbsAttr: number; // max absolute edge value
    maxAbsSubcompAct: number; // max absolute subcomponent activation for normalization
    l0_total: number; // total active components at current CI threshold
    optimization?: OptimizationResult;
};

/** Build edge indexes from flat edge array (single pass) */
export function buildEdgeIndexes(edges: Edge[]): {
    edgesBySource: Map<string, Edge[]>;
    edgesByTarget: Map<string, Edge[]>;
} {
    const edgesBySource = new Map<string, Edge[]>();
    const edgesByTarget = new Map<string, Edge[]>();

    for (const edge of edges) {
        const bySrc = edgesBySource.get(edge.src);
        if (bySrc) {
            bySrc.push(edge);
        } else {
            edgesBySource.set(edge.src, [edge]);
        }

        const byTgt = edgesByTarget.get(edge.tgt);
        if (byTgt) {
            byTgt.push(edge);
        } else {
            edgesByTarget.set(edge.tgt, [edge]);
        }
    }

    return { edgesBySource, edgesByTarget };
}

export type MaskType = "stochastic" | "ci";

export type OptimizationResult = {
    imp_min_coeff: number;
    steps: number;
    pnorm: number;
    beta: number;
    // CE loss params (optional - required together)
    label_token: number | null;
    label_str: string | null;
    ce_loss_coeff: number | null;
    label_prob: number | null;
    // KL loss param (optional)
    kl_loss_coeff: number | null;
    mask_type: MaskType;
};

export type ComponentSummary = {
    subcomponent_idx: number;
    mean_ci: number;
};

export type ActivationContextsSummary = Record<string, ComponentSummary[]>;

// Note: Token P/R/lift stats come from /token_stats endpoint (batch job), not here
export type ComponentDetail = {
    subcomponent_idx: number;
    mean_ci: number;
    example_tokens: string[][];
    example_ci: number[][];
    example_component_acts: number[][];
};

export type CorrelatedComponent = {
    component_key: string;
    score: number;
    count_i: number; // Subject (query component) firing count
    count_j: number; // Object (this component) firing count
    count_ij: number; // Co-occurrence count
    n_tokens: number; // Total tokens
};

export type ComponentCorrelations = {
    precision: CorrelatedComponent[];
    recall: CorrelatedComponent[];
    jaccard: CorrelatedComponent[];
    pmi: CorrelatedComponent[];
    bottom_pmi: CorrelatedComponent[];
};

// Token P/R/lift/PMI for a single category (input or output)
export type TokenPRLiftPMI = {
    top_recall: [string, number][]; // [(token, value), ...] sorted desc
    top_precision: [string, number][]; // [(token, value), ...] sorted desc
    top_lift: [string, number][]; // [(token, lift), ...] sorted desc
    top_pmi: [string, number][]; // [(token, pmi), ...] highest positive association
    bottom_pmi: [string, number][]; // [(token, pmi), ...] highest negative association
};

// Token stats from batch job - includes both input and output stats
export type TokenStats = {
    input: TokenPRLiftPMI; // What tokens activate this component
    output: TokenPRLiftPMI; // What tokens this component predicts
};

export type TokenizeResult = {
    token_ids: number[];
    tokens: string[];
    text: string;
};

export type TokenInfo = {
    id: number;
    string: string;
};

// Client-side computed types

export type LayerInfo = {
    name: string;
    block: number;
    type: "attn" | "mlp" | "embed" | "output";
    subtype: string;
};

export type NodePosition = {
    x: number;
    y: number;
};

export type PinnedNode = {
    layer: string;
    seqIdx: number;
    cIdx: number;
};

export type HoveredNode = {
    layer: string;
    seqIdx: number;
    cIdx: number;
};

export type HoveredEdge = {
    src: string;
    tgt: string;
    val: number;
};

// Graph layout result
export type LayoutResult = {
    nodePositions: Record<string, NodePosition>;
    layerYPositions: Record<string, number>;
    seqWidths: number[];
    seqXStarts: number[];
    width: number;
    height: number;
};

// Component probe result
export type ComponentProbeResult = {
    tokens: string[];
    ci_values: number[];
    subcomp_acts: number[];
};

// Display name mapping for special layers
const LAYER_DISPLAY_NAMES: Record<string, string> = {
    lm_head: "W_U",
};

/** Get display name for a layer (e.g., "lm_head" -> "W_U") */
export function getLayerDisplayName(layer: string): string {
    return LAYER_DISPLAY_NAMES[layer] ?? layer;
}

/** Format a node key for display, replacing layer names with display names */
export function formatNodeKeyForDisplay(nodeKey: string): string {
    const [layer, ...rest] = nodeKey.split(":");
    const displayName = getLayerDisplayName(layer);
    return [displayName, ...rest].join(":");
}

// Node intervention helpers
// "wte" and "output" are pseudo-layers used for visualization but are not part of the
// decomposed model. They cannot be intervened on - only the internal layers (attn/mlp)
// can have their components selectively activated.
const NON_INTERVENTABLE_LAYERS = new Set(["wte", "output"]);

export function isInterventableNode(nodeKey: string): boolean {
    const layer = nodeKey.split(":")[0];
    return !NON_INTERVENTABLE_LAYERS.has(layer);
}

export function filterInterventableNodes(nodeKeys: Iterable<string>): Set<string> {
    return new Set([...nodeKeys].filter(isInterventableNode));
}
