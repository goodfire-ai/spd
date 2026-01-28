import type { GraphData } from "../../lib/promptAttributionsTypes";
import type { InterventionRunSummary } from "../../lib/interventionTypes";
import type { NormalizeType } from "../../lib/api";

export type MaskType = "stochastic" | "ci";
export type LossType = "ce" | "kl";

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

export type CELossConfig = {
    type: "ce";
    coeff: number;
    position: number;
    labelTokenId: number;
    labelTokenText: string;
};

export type KLLossConfig = {
    type: "kl";
    coeff: number;
    position: number;
};

export type LossConfig = CELossConfig | KLLossConfig;

export type OptimizeConfig = {
    loss: LossConfig;
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

/** State for the draft prompt input */
export type DraftState = {
    text: string;
    tokenPreview: { tokens: string[]; loading: boolean };
    isAdding: boolean;
};

export function defaultDraftState(): DraftState {
    return {
        text: "",
        tokenPreview: { tokens: [], loading: false },
        isAdding: false,
    };
}

/** Discriminated union for the tab view - makes invalid states unrepresentable */
export type TabViewState =
    | { view: "draft"; draft: DraftState }
    | { view: "loading" }
    | { view: "card"; cardId: number }
    | { view: "error"; error: string };

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

export function defaultOptimizeConfig(numTokens: number): OptimizeConfig {
    return {
        loss: {
            type: "ce",
            coeff: 1,
            position: numTokens - 1,
            labelTokenId: -1, // Invalid - must be set before use
            labelTokenText: "",
        },
        impMinCoeff: 0.001,
        steps: 2000,
        pnorm: 0.3,
        beta: 0,
        maskType: "stochastic",
    };
}
