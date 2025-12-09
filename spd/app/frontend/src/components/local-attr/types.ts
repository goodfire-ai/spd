import type { SvelteSet } from "svelte/reactivity";
import type { GraphData } from "../../lib/localAttributionsTypes";
import type { InterventionRunSummary } from "../../lib/interventionTypes";

export type StoredGraph = {
    id: string;
    dbId: number; // database ID for API calls
    label: string;
    data: GraphData;
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
    activeGraphId: string | null;
    activeView: "graph" | "interventions";
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
