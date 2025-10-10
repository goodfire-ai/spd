import { writable } from "svelte/store";
import type {
    OutputTokenLogit,
    RunPromptResponse,
    MaskOverrideDTO,
    ComponentMask,
    LayerCIs,
    MatrixCausalImportances,
    AblationStats
} from "$lib/api";

export interface AblationResult {
    tokenLogits: OutputTokenLogit[][];
    applied_mask: ComponentMask;
    id: number;
    maskOverride?: MaskOverrideDTO;
    ablationStats: AblationStats;
}

export interface SelectedToken {
    layer: string;
    tokenIdx: number;
    token: string;
}

export interface LayerMask {
    layer: string;
    mask: number[];
    name?: string;
}

export interface CombinedMask {
    layer: string;
    tokenIndices: number[];
    description: string;
    l0: number;
    createdAt: number;
}

export interface PromptWorkspace {
    promptId: string;
    promptData: RunPromptResponse;
    ablationResults: AblationResult[];
    runAblation: ComponentMask;
}

export const ablationComponentMask = writable<ComponentMask>({});
export const ablationResults = writable<AblationResult[]>([]);
export const isScrolling = writable(false);

// Multi-select state for combining masks
export const multiSelectMode = writable(false);
export const selectedTokensForCombining = writable<SelectedToken[]>([]);
export const combinedMasks = writable<CombinedMask[]>([]);

// Multi-prompt workspace management
export const promptWorkspaces = writable<PromptWorkspace[]>([]);
export const currentWorkspaceIndex = writable<number>(0);
