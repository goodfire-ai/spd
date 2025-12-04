import type { GraphData } from "../../lib/localAttributionsTypes";

export type CachedGraph = {
    id: string;
    label: string;
    data: GraphData;
};

export type PromptCard = {
    id: string;
    promptId: number | null;
    tokens: string[];
    tokenIds: number[] | null;
    isCustom: boolean;
    graphs: CachedGraph[];
    activeGraphId: string | null;
};

export type OptimizeConfig = {
    labelTokenText: string;
    labelTokenId: number | null;
    labelTokenPreview: string | null;
    impMinCoeff: number;
    ceLossCoeff: number;
    steps: number;
    pnorm: number;
};

export type ComputeOptions = {
    maxMeanCI: number;
    normalizeEdges: boolean;
    useOptimized: boolean;
    optimizeConfig: OptimizeConfig;
};

export type LoadingStage = {
    name: string;
    progress: number | null; // 0-1, or null for indeterminate
};

export type LoadingState = {
    stages: LoadingStage[];
    currentStage: number; // 0-indexed
};
