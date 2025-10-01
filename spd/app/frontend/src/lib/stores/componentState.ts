import { writable } from "svelte/store";
import type { SubcomponentMask, OutputTokenLogit, SparseVector, RunPromptResponse, MaskOverrideDTO } from "$lib/api";



export interface PopupData {
    token: string;
    tokenIdx: number;
    layer: string;
    layerIdx: number;
    tokenCis: SparseVector;
}

export interface AblationResult {
    tokenLogits: OutputTokenLogit[][];
    applied_mask: SubcomponentMask;
    id: number;
    maskOverride?: MaskOverrideDTO;
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
    runAblation: SubcomponentMask;
}

export const ablationSubcomponentMask = writable<SubcomponentMask>({});
export const popupData = writable<PopupData | null>(null);
export const ablationResults = writable<AblationResult[]>([]);
export const isScrolling = writable(false);

// Multi-select state for combining masks
export const multiSelectMode = writable(false);
export const selectedTokensForCombining = writable<SelectedToken[]>([]);
export const combinedMasks = writable<CombinedMask[]>([]);

// Multi-prompt workspace management
export const promptWorkspaces = writable<PromptWorkspace[]>([]);
export const currentWorkspaceIndex = writable<number>(0);
