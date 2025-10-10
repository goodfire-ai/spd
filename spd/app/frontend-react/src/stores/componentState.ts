import { create } from 'zustand';
import type {
    OutputTokenLogit,
    RunPromptResponse,
    MaskOverrideDTO,
    ComponentMask,
    AblationStats
} from '../api';

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

interface ComponentState {
    ablationComponentMask: ComponentMask;
    ablationResults: AblationResult[];
    isScrolling: boolean;
    multiSelectMode: boolean;
    selectedTokensForCombining: SelectedToken[];
    combinedMasks: CombinedMask[];
    promptWorkspaces: PromptWorkspace[];
    currentWorkspaceIndex: number;

    setAblationComponentMask: (mask: ComponentMask) => void;
    setAblationResults: (results: AblationResult[]) => void;
    setIsScrolling: (scrolling: boolean) => void;
    setMultiSelectMode: (mode: boolean) => void;
    setSelectedTokensForCombining: (tokens: SelectedToken[]) => void;
    setCombinedMasks: (masks: CombinedMask[]) => void;
    setPromptWorkspaces: (workspaces: PromptWorkspace[]) => void;
    setCurrentWorkspaceIndex: (index: number) => void;
}

export const useComponentStore = create<ComponentState>((set) => ({
    ablationComponentMask: {},
    ablationResults: [],
    isScrolling: false,
    multiSelectMode: false,
    selectedTokensForCombining: [],
    combinedMasks: [],
    promptWorkspaces: [],
    currentWorkspaceIndex: 0,

    setAblationComponentMask: (mask) => set({ ablationComponentMask: mask }),
    setAblationResults: (results) => set({ ablationResults: results }),
    setIsScrolling: (scrolling) => set({ isScrolling: scrolling }),
    setMultiSelectMode: (mode) => set({ multiSelectMode: mode }),
    setSelectedTokensForCombining: (tokens) => set({ selectedTokensForCombining: tokens }),
    setCombinedMasks: (masks) => set({ combinedMasks: masks }),
    setPromptWorkspaces: (workspaces) => set({ promptWorkspaces: workspaces }),
    setCurrentWorkspaceIndex: (index) => set({ currentWorkspaceIndex: index }),
}));
