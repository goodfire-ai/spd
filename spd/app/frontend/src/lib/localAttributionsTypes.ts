import { SvelteSet } from "svelte/reactivity";

/** Types for the local attributions visualizer */

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

export type OutputProbEntry = {
    prob: number;
    token: string;
};

export type GraphData = {
    id: number;
    tokens: string[];
    edges: Edge[];
    outputProbs: Record<string, OutputProbEntry>; // key is "seq:cIdx"
    nodeImportance: Record<string, number>; // node key -> sum of squared edge values
    maxAbsAttr: number; // max absolute edge value
    l0_total: number; // total active components at current CI threshold
    optimization?: OptimizationResult;
};

export type OptimizationResult = {
    label_token: number;
    label_str: string;
    imp_min_coeff: number;
    ce_loss_coeff: number;
    steps: number;
    label_prob: number;
};

export type ComponentSummary = {
    subcomponent_idx: number;
    mean_ci: number;
};

export type ActivationContextsSummary = Record<string, ComponentSummary[]>;

export type ComponentDetail = {
    subcomponent_idx: number;
    mean_ci: number;
    example_tokens: string[][];
    example_ci: number[][];
    example_active_pos: number[];
    example_active_ci: number[];
    pr_tokens: string[];
    pr_recalls: number[];
    pr_precisions: number[];
    // TODO: Re-enable token uplift after performance optimization
    // predicted_tokens: string[];
    // predicted_lifts: number[];
    // predicted_firing_probs: number[];
    // predicted_base_probs: number[];
};

export type SearchResult = {
    query: { components: string[]; mode: string };
    count: number;
    results: PromptPreview[];
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
};

// Node intervention helpers
// "wte" and "output" are pseudo-layers used for visualization but are not part of the
// decomposed model. They cannot be intervened on - only the internal layers (attn/mlp)
// can have their components selectively activated.
const NON_INTERVENTABLE_LAYERS = new Set(["wte", "output"]);

export function isInterventableNode(nodeKey: string): boolean {
    const layer = nodeKey.split(":")[0];
    return !NON_INTERVENTABLE_LAYERS.has(layer);
}

export function filterInterventableNodes(nodeKeys: Iterable<string>): SvelteSet<string> {
    const result = new SvelteSet<string>();
    for (const key of nodeKeys) {
        if (isInterventableNode(key)) result.add(key);
    }
    return result;
}
