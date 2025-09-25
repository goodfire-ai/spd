import { writable } from "svelte/store";
import type { ComponentMask, OutputTokenLogit } from "$lib/api";

export interface PopupData {
    token: string;
    tokenIdx: number;
    layer: string;
    layerIdx: number;
    tokenCi: {
        l0: number;
        component_cis: number[];
        indices: number[];
    };
}

export interface AblationResult {
    tokenLogits: OutputTokenLogit[][];
    applied_mask: ComponentMask;
    id: number;
}

export const runAblation = writable<ComponentMask>({});
export const popupData = writable<PopupData | null>(null);
export const ablationResults = writable<AblationResult[]>([]);
export const isScrolling = writable(false);