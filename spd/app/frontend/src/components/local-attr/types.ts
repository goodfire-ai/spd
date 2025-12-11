import type { SvelteSet } from "svelte/reactivity";
import type { GraphData } from "../../lib/localAttributionsTypes";
import type { InterventionRunSummary } from "../../lib/interventionTypes";
import type { NormalizeType } from "../../lib/localAttributionsApi";

export type ViewSettings = {
    topK: number;
    nodeLayout: "importance" | "shuffled" | "jittered";
    componentGap: number;
    layerGap: number;
    normalizeEdges: NormalizeType;
    ciThreshold: number;
};

export type StoredGraph = {
    id: string;
    dbId: number; // database ID for API calls
    label: string;
    data: GraphData;
    viewSettings: ViewSettings;
    // Composer state for interventions
    composerSelection: SvelteSet<string>; // currently selected node keys
    interventionRuns: InterventionRunSummary[]; // persisted runs
    activeRunId: number | null; // which run is selected (for restoring selection)
};

export type PromptCard = {
    id: string;
    promptId: number;
    tokens: string[];
    tokenIds: number[];
    isCustom: boolean;
    graphs: StoredGraph[];
    activeGraphId: string | null; // null means "new graph" mode when graphs exist, or initial state
    activeView: "graph" | "interventions";
    // Config for creating new graphs (per-card, not shared globally)
    newGraphConfig: OptimizeConfig;
    useOptimized: boolean; // whether to compute optimized graph
};

export type OptimizeConfig = {
    // CE loss settings (active when ceLossCoeff > 0 AND labelTokenId is set)
    labelTokenText: string;
    labelTokenId: number | null;
    labelTokenPreview: string | null;
    ceLossCoeff: number;
    // KL loss settings (active when klLossCoeff > 0)
    klLossCoeff: number;
    // Common settings
    impMinCoeff: number;
    steps: number;
    pnorm: number;
};

export type ComputeOptions = {
    ciThreshold: number;
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

export function defaultOptimizeConfig(): OptimizeConfig {
    return {
        labelTokenText: "",
        labelTokenId: null,
        labelTokenPreview: null,
        ceLossCoeff: 0,
        klLossCoeff: 0,
        impMinCoeff: 0.1,
        steps: 2000,
        pnorm: 0.3,
    };
}
