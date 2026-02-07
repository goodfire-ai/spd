/** Types for the intervention forward pass feature */

export type InterventionNode = {
    layer: string;
    seq_pos: number;
    component_idx: number;
};

export type TokenPrediction = {
    token: string;
    token_id: number;
    spd_prob: number;
    target_prob: number;
    logit: number;
    target_logit: number;
};

export type InterventionResponse = {
    input_tokens: string[];
    predictions_per_position: TokenPrediction[][];
};

/** A forked intervention run with modified tokens */
export type ForkedInterventionRunSummary = {
    id: number;
    token_replacements: [number, number][]; // [(seq_pos, new_token_id), ...]
    result: InterventionResponse;
    created_at: string;
};

/** Persisted intervention run from the server */
export type InterventionRunSummary = {
    id: number;
    selected_nodes: string[]; // node keys (layer:seq:cIdx)
    result: InterventionResponse;
    created_at: string;
    forked_runs?: ForkedInterventionRunSummary[]; // child runs with modified tokens
};

/** Request to run and save an intervention */
export type RunInterventionRequest = {
    graph_id: number;
    text: string;
    selected_nodes: string[];
    top_k?: number;
};
