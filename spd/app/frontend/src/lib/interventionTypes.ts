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
};

export type InterventionResponse = {
    input_tokens: string[];
    predictions_per_position: TokenPrediction[][];
};
