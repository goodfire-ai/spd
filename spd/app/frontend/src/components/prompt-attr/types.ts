import type { GraphData } from "../../lib/promptAttributionsTypes";
import type { InterventionRunSummary } from "../../lib/interventionTypes";
import type { NormalizeType } from "../../lib/api";

export type MaskType = "stochastic" | "ci";

export type ViewSettings = {
    topK: number;
    componentGap: number;
    layerGap: number;
    normalizeEdges: NormalizeType;
    ciThreshold: number;
};

/** Persisted graph data from the database */
export type StoredGraph = {
    id: number; // database ID
    label: string;
    data: GraphData;
    viewSettings: ViewSettings;
    interventionRuns: InterventionRunSummary[];
};

/** Transient UI state for the intervention composer, keyed by graph ID */
export type ComposerState = {
    selection: Set<string>; // currently selected node keys
    activeRunId: number | null; // which run is selected (for restoring selection)
};

export type PromptCard = {
    id: number; // database prompt ID
    tokens: string[];
    tokenIds: number[];
    isCustom: boolean;
    graphs: StoredGraph[];
    activeGraphId: number | null; // null means "new graph" mode when graphs exist, or initial state
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
    beta: number;
    maskType: MaskType;
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

/** Generic state for async actions without a meaningful result */
export type ActionState = { status: "idle" } | { status: "loading" } | { status: "error"; error: string };

/** State for graph computation - tracks which card is computing, progress, and errors */
export type GraphComputeState =
    | { status: "idle" }
    | { status: "computing"; cardId: number; progress: LoadingState }
    | { status: "error"; error: string };

/** State for prompt generation - tracks progress and count */
export type PromptGenerateState =
    | { status: "idle" }
    | { status: "generating"; progress: number; count: number }
    | { status: "error"; error: string };

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
        beta: 0,
        maskType: "stochastic",
    };
}
