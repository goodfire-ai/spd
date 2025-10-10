import React, { useState, useEffect } from 'react';
import type {
    ClusterRunDTO,
    RunPromptResponse,
    ComponentMask,
    MatrixCausalImportances,
    AvailablePrompt,
    ClusterDashboardResponse
} from '../api';
import * as api from '../api';
import { useComponentStore, type PromptWorkspace, type AblationResult } from '../stores/componentState';
import { ComponentHeatmap } from './ComponentHeatmap';
import { DisabledComponentsPanel } from './DisabledComponentsPanel';
import { OriginalPredictions } from './OriginalPredictions';
import { AblationPredictions } from './AblationPredictions';
import { SavedMasksPanel } from './SavedMasksPanel';
import { ComponentDetailModal } from './ComponentDetailModal';

interface InterventionsTabProps {
    cluster_run: ClusterRunDTO;
    iteration: number;
}

interface PopupData {
    token: string;
    tokenIdx: number;
    layerIdx: number;
    layerName: string;
    tokenCIs: MatrixCausalImportances;
}

export const InterventionsTab: React.FC<InterventionsTabProps> = ({ cluster_run, iteration }) => {
    const {
        ablationComponentMask,
        setAblationComponentMask,
        ablationResults,
        setAblationResults,
        promptWorkspaces,
        setPromptWorkspaces,
        currentWorkspaceIndex,
        setCurrentWorkspaceIndex,
    } = useComponentStore();

    const [isLoading, setIsLoading] = useState(false);
    const [result, setResult] = useState<RunPromptResponse | null>(null);
    const [currentPromptId, setCurrentPromptId] = useState<string | null>(null);
    const [availablePrompts, setAvailablePrompts] = useState<AvailablePrompt[] | null>(null);
    const [showAvailablePrompts, setShowAvailablePrompts] = useState(false);
    const [currentAblationPage, setCurrentAblationPage] = useState(0);
    const [popupData, setPopupData] = useState<PopupData | null>(null);
    const [dashboard, setDashboard] = useState<ClusterDashboardResponse | null>(null);

    useEffect(() => {
        loadAvailablePrompts();
        loadDashboard();
    }, []);

    const loadAvailablePrompts = async () => {
        try {
            const prompts = await api.getAvailablePrompts();
            setAvailablePrompts(prompts);
        } catch (error: any) {
            console.error("Failed to load prompts:", error.message);
        }
    };

    const loadDashboard = async () => {
        const dashboardData = await api.getClusterDashboardData({
            iteration,
            n_samples: 16,
            n_batches: 2,
            batch_size: 64,
            context_length: 64,
        });
        setDashboard(dashboardData);
    };

    const toggleAvailablePrompts = () => {
        setShowAvailablePrompts(!showAvailablePrompts);
    };

    const createNewWorkspace = (promptData: RunPromptResponse): PromptWorkspace => {
        const newMask: ComponentMask = {};
        for (const layer of promptData.layer_cis) {
            newMask[layer.module] = promptData.prompt_tokens.map(() => []);
        }

        return {
            promptId: promptData.prompt_id,
            promptData,
            ablationResults: [],
            runAblation: newMask,
        };
    };

    const switchToWorkspace = (index: number) => {
        if (index >= 0 && index < promptWorkspaces.length) {
            setCurrentWorkspaceIndex(index);
            const workspace = promptWorkspaces[index];
            setResult(workspace.promptData);
            setCurrentPromptId(workspace.promptId);
            setAblationComponentMask(workspace.runAblation);
            setAblationResults(workspace.ablationResults);
        }
    };

    const closeWorkspace = (index: number) => {
        const newWorkspaces = promptWorkspaces.filter((_, i) => i !== index);
        setPromptWorkspaces(newWorkspaces);

        if (newWorkspaces.length === 0) {
            setResult(null);
            setCurrentPromptId(null);
            setCurrentWorkspaceIndex(0);
        } else if (index <= currentWorkspaceIndex) {
            const newIndex = Math.max(0, currentWorkspaceIndex - 1);
            setCurrentWorkspaceIndex(newIndex);
            switchToWorkspace(newIndex);
        }
    };

    const runPromptByIndex = async (datasetIndex: number) => {
        setIsLoading(true);
        try {
            const promptData = await api.runPromptByIndex(datasetIndex);
            const newWorkspace = createNewWorkspace(promptData);

            const newWorkspaces = [...promptWorkspaces, newWorkspace];
            setPromptWorkspaces(newWorkspaces);
            setCurrentWorkspaceIndex(newWorkspaces.length - 1);
            switchToWorkspace(newWorkspaces.length - 1);
        } catch (error: any) {
            console.error(`Error: ${error.message}`);
        }
        setIsLoading(false);
    };

    const applyMaskAsAblation = async (maskId: string) => {
        if (!result || !currentPromptId) return;

        setIsLoading(true);
        try {
            const maskResult = await api.applyMaskAsAblation(currentPromptId, maskId);
            const masks = await api.getMaskOverrides();
            const appliedMask = masks.find((m) => m.id === maskId);

            const newResults: AblationResult[] = [
                ...ablationResults,
                {
                    tokenLogits: maskResult.token_logits,
                    applied_mask: {},
                    id: Date.now(),
                    maskOverride: appliedMask,
                    ablationStats: maskResult.ablation_stats,
                },
            ];
            setAblationResults(newResults);
        } catch (error: any) {
            console.error(`Error: ${error.message}`);
        }
        setIsLoading(false);
    };

    const initializeRunAblation = () => {
        if (!result) return;
        const newMask: ComponentMask = {};
        for (const layer of result.layer_cis) {
            newMask[layer.module] = result.prompt_tokens.map(() => []);
        }
        setAblationComponentMask(newMask);
    };

    const toggleComponentDisabled = (layerName: string, tokenIdx: number, componentIdx: number) => {
        const newMask = { ...ablationComponentMask };
        if (!newMask[layerName]) {
            newMask[layerName] = result!.prompt_tokens.map(() => []);
        }

        const disabledComponents = newMask[layerName][tokenIdx];
        const existingIdx = disabledComponents.indexOf(componentIdx);

        if (existingIdx === -1) {
            disabledComponents.push(componentIdx);
        } else {
            disabledComponents.splice(existingIdx, 1);
        }

        setAblationComponentMask(newMask);
    };

    const isComponentDisabled = (layerName: string, tokenIdx: number, componentIdx: number): boolean => {
        return ablationComponentMask[layerName]?.[tokenIdx]?.includes(componentIdx) || false;
    };

    const sendAblation = async () => {
        if (!result || !currentPromptId) return;

        setIsLoading(true);
        try {
            const data = await api.ablateComponents(currentPromptId, ablationComponentMask);

            const deepCopyMask: ComponentMask = {};
            for (const [layerName, tokenArrays] of Object.entries(ablationComponentMask)) {
                deepCopyMask[layerName] = tokenArrays.map((tokenMask) => [...tokenMask]);
            }

            const newResults: AblationResult[] = [
                ...ablationResults,
                {
                    tokenLogits: data.token_logits,
                    applied_mask: deepCopyMask,
                    id: Date.now(),
                    ablationStats: data.ablation_stats,
                },
            ];
            setAblationResults(newResults);
        } catch (error: any) {
            console.error(`Error: ${error.message}`);
        }
        setIsLoading(false);
    };

    const runRandomPrompt = async () => {
        if (!availablePrompts) {
            console.error("No prompts available");
            return;
        }
        runPromptByIndex(Math.floor(Math.random() * availablePrompts.length));
    };

    const openPopup = (
        token: string,
        tokenIdx: number,
        layerIdx: number,
        layerName: string,
        tokenCIs: MatrixCausalImportances
    ) => {
        setPopupData({ token, tokenIdx, layerIdx, layerName, tokenCIs });
    };

    const closePopup = () => {
        setPopupData(null);
    };

    useEffect(() => {
        if (result) {
            initializeRunAblation();
        }
    }, [result]);

    return (
        <div className="tab-content">
            <SavedMasksPanel onApplyMask={applyMaskAsAblation} />

            <div className="workspace-navigation">
                <div className="workspace-header">
                    <h3>Prompt Workspaces</h3>
                    {showAvailablePrompts ? (
                        <button className="add-prompt-btn" onClick={toggleAvailablePrompts}>
                            Cancel
                        </button>
                    ) : (
                        <>
                            <button className="add-prompt-btn" onClick={runRandomPrompt}>
                                + Random Prompt
                            </button>
                            <button className="add-prompt-btn" onClick={toggleAvailablePrompts}>
                                + Add Prompt
                            </button>
                        </>
                    )}
                </div>

                {showAvailablePrompts && (
                    <div className="available-prompts-dropdown">
                        {availablePrompts == null ? (
                            <p>Loading prompts...</p>
                        ) : (
                            <div className="prompt-list">
                                {availablePrompts.map((prompt, i) => (
                                    <button
                                        key={prompt.index}
                                        className="prompt-button"
                                        onClick={() => {
                                            runPromptByIndex(prompt.index);
                                            setShowAvailablePrompts(false);
                                        }}
                                        disabled={isLoading}
                                    >
                                        <span className="prompt-number">#{i + 1}</span>
                                        <span className="prompt-text">
                                            {prompt.full_text.slice(0, 40)}
                                            {prompt.full_text.length > 40 ? "..." : ""}
                                        </span>
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>
                )}

                {promptWorkspaces.length > 0 ? (
                    <div className="workspace-list">
                        {promptWorkspaces.map((workspace, i) => (
                            <button
                                key={i}
                                className={`workspace-item ${i === currentWorkspaceIndex ? "active" : ""}`}
                                onClick={() => switchToWorkspace(i)}
                            >
                                <span className="workspace-number">#{i + 1}</span>
                                <span className="workspace-text">
                                    {workspace.promptData.prompt_tokens.slice(0, 8).join(" ")}...
                                </span>
                                <span
                                    className="workspace-close"
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        closeWorkspace(i);
                                    }}
                                >
                                    ×
                                </span>
                            </button>
                        ))}
                    </div>
                ) : (
                    <div className="empty-workspaces">No prompts loaded. Click "Add Prompt" to start.</div>
                )}
            </div>

            <div className="main-layout">
                <div className="left-panel">
                    {result && currentPromptId && (
                        <>
                            <ComponentHeatmap
                                result={result}
                                promptId={currentPromptId}
                                onCellPopop={openPopup}
                            />
                            <DisabledComponentsPanel
                                promptTokens={result.prompt_tokens}
                                isLoading={isLoading}
                                onSendAblation={sendAblation}
                                onToggleComponent={toggleComponentDisabled}
                            />
                        </>
                    )}
                </div>

                <div className="right-panel">
                    {result && result.full_run_token_logits && (
                        <OriginalPredictions
                            tokenLogits={result.full_run_token_logits}
                            promptTokens={result.prompt_tokens}
                            title="Original Model Predictions"
                        />
                    )}

                    {result && result.ci_masked_token_logits && (
                        <OriginalPredictions
                            tokenLogits={result.ci_masked_token_logits}
                            promptTokens={result.prompt_tokens}
                            title="Original <strong>CI Masked</strong> Model Predictions"
                        />
                    )}

                    {result && ablationResults.length > 0 && (
                        <div className="ablation-results-container">
                            <div className="pagination-header">
                                <h3>Ablation Results ({ablationResults.length} total)</h3>
                                <div className="pagination-controls">
                                    <button
                                        onClick={() => setCurrentAblationPage(Math.max(0, currentAblationPage - 1))}
                                        disabled={currentAblationPage === 0}
                                    >
                                        ←
                                    </button>
                                    <span>
                                        {currentAblationPage + 1} / {ablationResults.length}
                                    </span>
                                    <button
                                        onClick={() =>
                                            setCurrentAblationPage(
                                                Math.min(ablationResults.length - 1, currentAblationPage + 1)
                                            )
                                        }
                                        disabled={currentAblationPage === ablationResults.length - 1}
                                    >
                                        →
                                    </button>
                                </div>
                            </div>
                            <AblationPredictions
                                tokenLogits={ablationResults[currentAblationPage].tokenLogits}
                                promptTokens={result.prompt_tokens}
                                appliedMask={ablationResults[currentAblationPage].applied_mask}
                                maskOverride={ablationResults[currentAblationPage].maskOverride}
                                ablationStats={ablationResults[currentAblationPage].ablationStats}
                            />
                        </div>
                    )}
                </div>
            </div>

            {popupData && dashboard && (
                <ComponentDetailModal
                    cluster={cluster_run}
                    popupData={popupData}
                    dashboard={dashboard}
                    onClose={closePopup}
                    toggleComponent={toggleComponentDisabled}
                    isComponentDisabled={isComponentDisabled}
                />
            )}
        </div>
    );
};
