// Type definitions for local attributions data

export type SparseAttribution = [number, number, number, number, number] | [number, number, number, number];

export type PairData = {
  source: string;
  target: string;
  is_cross_seq: boolean;
  attribution: SparseAttribution[];
  trimmed_c_in_idxs: number[];
  trimmed_c_out_idxs: number[];
};

export type SubcomponentData = {
  subcomponent_idx: number;
  mean_ci: number;
  example_tokens: string[][];
  example_ci: number[][];
  example_active_pos: number[];
  example_active_ci: number[];
  pr_tokens?: string[];
  pr_precisions?: number[];
  predicted_tokens?: string[];
  predicted_probs?: number[];
};

export type LocalAttributionsData = {
  tokens: string[];
  pairs: PairData[];
  activation_contexts?: Record<string, SubcomponentData[]>;
};

export type LayerInfo = {
  name: string;
  block: number;
  type: 'attn' | 'mlp';
  subtype: string;
};

export type EdgeData = {
  srcKey: string;
  tgtKey: string;
  val: number;
};

export type NodePosition = {
  x: number;
  y: number;
};

export type PinnedNode = {
  layer: string;
  cIdx: number;
  nodeKey: string;
};
