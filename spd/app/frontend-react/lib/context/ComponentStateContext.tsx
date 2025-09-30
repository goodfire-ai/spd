"use client";

import React, { createContext, useContext, useState, ReactNode } from "react";
import type { ComponentMask, OutputTokenLogit, SparseVector, RunPromptResponse, MaskOverrideDTO } from "../api";

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
    maskOverride?: MaskOverrideDTO;
}

export interface SelectedToken {
    layer: string;
    tokenIdx: number;
    token: string;
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

interface ComponentStateContextType {
    runAblation: ComponentMask;
    setRunAblation: (mask: ComponentMask) => void;
    popupData: PopupData | null;
    setPopupData: (data: PopupData | null) => void;
    ablationResults: AblationResult[];
    setAblationResults: (results: AblationResult[]) => void;
    multiSelectMode: boolean;
    setMultiSelectMode: (mode: boolean) => void;
    selectedTokensForCombining: SelectedToken[];
    setSelectedTokensForCombining: (tokens: SelectedToken[]) => void;
    combinedMasks: CombinedMask[];
    setCombinedMasks: (masks: CombinedMask[]) => void;
    promptWorkspaces: PromptWorkspace[];
    setPromptWorkspaces: (workspaces: PromptWorkspace[]) => void;
    currentWorkspaceIndex: number;
    setCurrentWorkspaceIndex: (index: number) => void;
}

const ComponentStateContext = createContext<ComponentStateContextType | undefined>(undefined);

export function ComponentStateProvider({ children }: { children: ReactNode }) {
    const [runAblation, setRunAblation] = useState<ComponentMask>({});
    const [popupData, setPopupData] = useState<PopupData | null>(null);
    const [ablationResults, setAblationResults] = useState<AblationResult[]>([]);
    const [multiSelectMode, setMultiSelectMode] = useState(false);
    const [selectedTokensForCombining, setSelectedTokensForCombining] = useState<SelectedToken[]>([]);
    const [combinedMasks, setCombinedMasks] = useState<CombinedMask[]>([]);
    const [promptWorkspaces, setPromptWorkspaces] = useState<PromptWorkspace[]>([]);
    const [currentWorkspaceIndex, setCurrentWorkspaceIndex] = useState(0);

    return (
        <ComponentStateContext.Provider
            value={{
                runAblation,
                setRunAblation,
                popupData,
                setPopupData,
                ablationResults,
                setAblationResults,
                multiSelectMode,
                setMultiSelectMode,
                selectedTokensForCombining,
                setSelectedTokensForCombining,
                combinedMasks,
                setCombinedMasks,
                promptWorkspaces,
                setPromptWorkspaces,
                currentWorkspaceIndex,
                setCurrentWorkspaceIndex
            }}
        >
            {children}
        </ComponentStateContext.Provider>
    );
}

export function useComponentState() {
    const context = useContext(ComponentStateContext);
    if (context === undefined) {
        throw new Error("useComponentState must be used within a ComponentStateProvider");
    }
    return context;
}