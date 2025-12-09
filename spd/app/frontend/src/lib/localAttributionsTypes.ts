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
    optimization?: OptimizationResult;
};

export type OptimizationResult = {
    label_token: number;
    label_str: string;
    imp_min_coeff: number;
    ce_loss_coeff: number;
    steps: number;
    label_prob: number;
    l0_total: number;
    l0_per_layer: Record<string, number>;
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
    predicted_tokens: string[];
    predicted_lifts: number[];
    predicted_firing_probs: number[];
    predicted_base_probs: number[];
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
