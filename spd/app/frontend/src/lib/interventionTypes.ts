/** Types for the intervention forward pass feature */

export type InterventionNode = {
    layer: string;
    seq_pos: number;
    component_idx: number;
};

export type TokenPrediction = {
    token: string;
    token_id: number;
    prob: number;
    logit: number;
    target_prob: number;
};

export type InterventionResponse = {
    input_tokens: string[];
    predictions_per_position: TokenPrediction[][];
};

/** Persisted intervention run from the server */
export type InterventionRunSummary = {
    id: number;
    selected_nodes: string[]; // node keys (layer:seq:cIdx)
    result: InterventionResponse;
    created_at: string;
};

/** Request to run and save an intervention */
export type RunInterventionRequest = {
    graph_id: number;
    text: string;
    selected_nodes: string[];
    top_k?: number;
};
