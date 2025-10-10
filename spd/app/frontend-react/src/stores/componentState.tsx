import React, { createContext, useContext, useState, ReactNode } from 'react';
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

const ComponentStateContext = createContext<ComponentState | undefined>(undefined);

export const ComponentStateProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const [ablationComponentMask, setAblationComponentMask] = useState<ComponentMask>({});
    const [ablationResults, setAblationResults] = useState<AblationResult[]>([]);
    const [isScrolling, setIsScrolling] = useState(false);
    const [multiSelectMode, setMultiSelectMode] = useState(false);
    const [selectedTokensForCombining, setSelectedTokensForCombining] = useState<SelectedToken[]>([]);
    const [combinedMasks, setCombinedMasks] = useState<CombinedMask[]>([]);
    const [promptWorkspaces, setPromptWorkspaces] = useState<PromptWorkspace[]>([]);
    const [currentWorkspaceIndex, setCurrentWorkspaceIndex] = useState(0);

    const value: ComponentState = {
        ablationComponentMask,
        ablationResults,
        isScrolling,
        multiSelectMode,
        selectedTokensForCombining,
        combinedMasks,
        promptWorkspaces,
        currentWorkspaceIndex,
        setAblationComponentMask,
        setAblationResults,
        setIsScrolling,
        setMultiSelectMode,
        setSelectedTokensForCombining,
        setCombinedMasks,
        setPromptWorkspaces,
        setCurrentWorkspaceIndex,
    };

    return <ComponentStateContext.Provider value={value}>{children}</ComponentStateContext.Provider>;
};

export const useComponentStore = () => {
    const context = useContext(ComponentStateContext);
    if (context === undefined) {
        throw new Error('useComponentStore must be used within a ComponentStateProvider');
    }
    return context;
};
