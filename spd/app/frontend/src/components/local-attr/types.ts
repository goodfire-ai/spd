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

export type ComputeOptions = {
    maxMeanCI: number;
    normalizeEdges: boolean;
    useOptimized: boolean;
    labelTokenText: string;
    labelTokenId: number | null;
    labelTokenPreview: string | null;
    impMinCoeff: number;
    ceLossCoeff: number;
    optimSteps: number;
    optimPnorm: number;
};

export type LoadingProgress = {
    current: number;
    total: number;
    stage: string;
};
