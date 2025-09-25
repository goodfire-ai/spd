import { writable } from "svelte/store";
import type { ComponentMask, OutputTokenLogit, SparseVector } from "$lib/api";



export interface PopupData {
    token: string;
    tokenIdx: number;
    layer: string;
    layerIdx: number;
    tokenCis: SparseVector;
}

export interface AblationResult {
    tokenLogits: OutputTokenLogit[][];
    applied_mask: ComponentMask;
    id: number;
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

export const runAblation = writable<ComponentMask>({});
export const popupData = writable<PopupData | null>(null);
export const ablationResults = writable<AblationResult[]>([]);
export const isScrolling = writable(false);

// Multi-select state for combining masks
export const multiSelectMode = writable(false);
export const selectedTokensForCombining = writable<SelectedToken[]>([]);
export const combinedMasks = writable<CombinedMask[]>([]);
