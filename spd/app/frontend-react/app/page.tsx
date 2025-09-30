"use client";

import { useState, useEffect, useRef } from "react";
import { api } from "@/lib/api";
import type { RunPromptResponse, ComponentMask, StatusDTO } from "@/lib/api";
import { useComponentState, type PromptWorkspace, type AblationResult } from "@/lib/context/ComponentStateContext";
import RunSelector from "@/components/RunSelector";
import ComponentHeatmap from "@/components/ComponentHeatmap";
import DisabledComponentsPanel from "@/components/DisabledComponentsPanel";
import ComponentDetailModal from "@/components/ComponentDetailModal";
import OriginalPredictions from "@/components/OriginalPredictions";
import AblationPredictions from "@/components/AblationPredictions";
import SavedMasksPanel, { type SavedMasksPanelRef } from "@/components/SavedMasksPanel";

export default function Home() {
    const {
        runAblation,
        setRunAblation,
        setPopupData,
        ablationResults,
        setAblationResults,
        promptWorkspaces,
        setPromptWorkspaces,
        currentWorkspaceIndex,
        setCurrentWorkspaceIndex
    } = useComponentState();

    const [isLoading, setIsLoading] = useState(false);
    const [result, setResult] = useState<RunPromptResponse | null>(null);
    const [currentPromptId, setCurrentPromptId] = useState<string | null>(null);
    const [wandbRunId, setWandbRunId] = useState<string | null>(null);
    const [loadingRun, setLoadingRun] = useState(false);
    const [availablePrompts, setAvailablePrompts] = useState<{ index: number; text: string; full_text: string }[]>([]);
    const [showAvailablePrompts, setShowAvailablePrompts] = useState(false);
    const savedMasksPanelRef = useRef<SavedMasksPanelRef>(null);

    useEffect(() => {
        getStatus();
        loadAvailablePrompts();
    }, []);

    const getStatus = async () => {
        const status = await api.getStatus();
        setWandbRunId(status.run_id);
    };

    const loadAvailablePrompts = async () => {
        try {
            const prompts = await api.getAvailablePrompts();
            setAvailablePrompts(prompts);
        } catch (error: any) {
            console.error("Failed to load prompts:", error.message);
        }
    };

    const toggleAvailablePrompts = () => {
        setShowAvailablePrompts(!showAvailablePrompts);
        if (!showAvailablePrompts && availablePrompts.length === 0) {
            loadAvailablePrompts();
        }
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
            runAblation: newMask
        };
    };

    const switchToWorkspace = (index: number) => {
        if (index >= 0 && index < promptWorkspaces.length) {
            setCurrentWorkspaceIndex(index);
            const workspace = promptWorkspaces[index];
            setResult(workspace.promptData);
            setCurrentPromptId(workspace.promptId);
            setRunAblation(workspace.runAblation);
            setAblationResults(workspace.ablationResults);
        }
    };

    const updateCurrentWorkspace = () => {
        if (currentWorkspaceIndex >= 0 && currentWorkspaceIndex < promptWorkspaces.length && result) {
            const updatedWorkspaces = [...promptWorkspaces];
            updatedWorkspaces[currentWorkspaceIndex] = {
                ...updatedWorkspaces[currentWorkspaceIndex],
                runAblation,
                ablationResults
            };
            setPromptWorkspaces(updatedWorkspaces);
        }
    };

    useEffect(() => {
        updateCurrentWorkspace();
    }, [runAblation, ablationResults]);

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

            setPromptWorkspaces([...promptWorkspaces, newWorkspace]);
            const newIndex = promptWorkspaces.length;
            setCurrentWorkspaceIndex(newIndex);
            switchToWorkspace(newIndex);
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

            setAblationResults([
                ...ablationResults,
                {
                    tokenLogits: maskResult.token_logits,
                    applied_mask: {},
                    id: Date.now(),
                    maskOverride: appliedMask
                }
            ]);
        } catch (error: any) {
            console.error(`Error: ${error.message}`);
        }
        setIsLoading(false);
    };

    const refreshSavedMasks = () => {
        if (savedMasksPanelRef.current) {
            savedMasksPanelRef.current.loadMasks();
        }
    };

    const initializeRunAblation = () => {
        if (!result) return;
        const newMask: ComponentMask = {};
        for (const layer of result.layer_cis) {
            newMask[layer.module] = result.prompt_tokens.map(() => []);
        }
        setRunAblation(newMask);
    };

    useEffect(() => {
        if (result) {
            initializeRunAblation();
        }
    }, [result]);

    const toggleComponentDisabled = (layerName: string, tokenIdx: number, componentIdx: number) => {
        if (!runAblation[layerName]) {
            runAblation[layerName] = result!.prompt_tokens.map(() => []);
        }

        const disabledComponents = runAblation[layerName][tokenIdx];
        const existingIdx = disabledComponents.indexOf(componentIdx);

        if (existingIdx === -1) {
            disabledComponents.push(componentIdx);
        } else {
            disabledComponents.splice(existingIdx, 1);
        }

        setRunAblation({ ...runAblation });
    };

    const isComponentDisabled = (layerName: string, tokenIdx: number, componentIdx: number): boolean => {
        return runAblation[layerName]?.[tokenIdx]?.includes(componentIdx) ?? false;
    };

    const sendAblation = async () => {
        if (!result || !currentPromptId) return;

        setIsLoading(true);
        try {
            const data = await api.ablateComponents(currentPromptId, runAblation);

            const deepCopyMask: ComponentMask = {};
            for (const [layerName, tokenArrays] of Object.entries(runAblation)) {
                deepCopyMask[layerName] = tokenArrays.map((tokenMask) => [...tokenMask]);
            }

            setAblationResults([
                ...ablationResults,
                {
                    tokenLogits: data.token_logits,
                    applied_mask: deepCopyMask,
                    id: Date.now()
                }
            ]);
        } catch (error: any) {
            console.error(`Error: ${error.message}`);
        }
        setIsLoading(false);
    };

    const openPopup = (token: string, tokenIdx: number, layer: string, layerIdx: number, tokenCis: any) => {
        setPopupData({ token, tokenIdx, layer, layerIdx, tokenCis });
    };

    const closePopup = () => {
        setPopupData(null);
    };

    return (
        <main className="p-2 font-sans">
            <div className="max-w-none m-0 flex flex-col gap-4">
                <div className="flex flex-col gap-4 p-4 bg-gray-50 rounded-lg border border-gray-300">
                    <RunSelector
                        wandbRunId={wandbRunId}
                        setWandbRunId={setWandbRunId}
                        loadingRun={loadingRun}
                        setLoadingRun={setLoadingRun}
                        isLoading={isLoading}
                    />

                    <div className="mb-4 p-4 bg-blue-50 border border-blue-300 rounded-lg">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="text-lg text-gray-800">Prompt Workspaces</h3>
                            <button
                                className="px-4 py-2 bg-green-500 text-white rounded font-medium hover:bg-green-600"
                                onClick={toggleAvailablePrompts}
                            >
                                {showAvailablePrompts ? "Cancel" : "+ Add Prompt"}
                            </button>
                        </div>

                        {showAvailablePrompts && (
                            <div className="mb-4 p-4 bg-white border border-gray-300 rounded-md shadow-md">
                                {availablePrompts.length === 0 ? (
                                    <p>Loading prompts...</p>
                                ) : (
                                    <div className="flex flex-col gap-2 max-h-[300px] overflow-y-auto">
                                        {availablePrompts.map((prompt, i) => (
                                            <button
                                                key={prompt.index}
                                                className="flex items-start gap-3 p-3 bg-white border-2 border-gray-200 rounded-md cursor-pointer text-left transition-all hover:border-blue-500 hover:bg-blue-50 disabled:opacity-60 disabled:cursor-not-allowed"
                                                onClick={() => {
                                                    runPromptByIndex(prompt.index);
                                                    setShowAvailablePrompts(false);
                                                }}
                                                disabled={isLoading}
                                            >
                                                <span className="font-bold text-gray-600 min-w-[2rem] mt-0.5">
                                                    #{i + 1}
                                                </span>
                                                <span className="flex-1 text-sm leading-relaxed text-gray-800">
                                                    {prompt.text}
                                                </span>
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}

                        {promptWorkspaces.length > 0 ? (
                            <div className="flex flex-col gap-2">
                                {promptWorkspaces.map((workspace, i) => (
                                    <button
                                        key={i}
                                        className={`flex items-center gap-3 p-3 border-2 rounded-md cursor-pointer text-left transition-all ${
                                            i === currentWorkspaceIndex
                                                ? "bg-blue-500 text-white border-blue-700"
                                                : "bg-white border-blue-300 hover:bg-blue-50 hover:border-blue-500"
                                        }`}
                                        onClick={() => switchToWorkspace(i)}
                                    >
                                        <span
                                            className={`font-bold min-w-[2rem] ${
                                                i === currentWorkspaceIndex ? "text-white" : "text-gray-600"
                                            }`}
                                        >
                                            #{i + 1}
                                        </span>
                                        <span className="flex-1 text-sm leading-relaxed overflow-hidden text-ellipsis whitespace-nowrap">
                                            {workspace.promptData.prompt_tokens.slice(0, 8).join(" ")}...
                                        </span>
                                        <span
                                            className={`text-xl font-bold cursor-pointer px-1 py-0.5 rounded-full transition-colors ${
                                                i === currentWorkspaceIndex
                                                    ? "text-white hover:bg-white/20"
                                                    : "text-gray-600 hover:bg-red-100 hover:text-red-600"
                                            }`}
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
                            <div className="py-8 text-center text-gray-600 italic">
                                No prompts loaded. Click &quot;Add Prompt&quot; to start.
                            </div>
                        )}
                    </div>

                    <SavedMasksPanel ref={savedMasksPanelRef} onApplyMask={applyMaskAsAblation} />
                </div>

                <div className="flex gap-4 min-h-[70vh]">
                    <div className="p-2 flex-[0_0_50%] max-w-[50%] sticky top-4 self-start max-h-[calc(100vh-2rem)] overflow-y-auto">
                        {result && currentPromptId && (
                            <>
                                <ComponentHeatmap
                                    result={result}
                                    promptId={currentPromptId}
                                    onCellClick={openPopup}
                                    onMaskCreated={refreshSavedMasks}
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

                    <div className="flex-1 overflow-y-auto pr-4">
                        {result?.full_run_token_logits && (
                            <OriginalPredictions
                                tokenLogits={result.full_run_token_logits}
                                promptTokens={result.prompt_tokens}
                                title="Original Model Predictions"
                            />
                        )}

                        {result?.ci_masked_token_logits && (
                            <OriginalPredictions
                                tokenLogits={result.ci_masked_token_logits}
                                promptTokens={result.prompt_tokens}
                                title="Original <strong>CI Masked</strong> Model Predictions"
                            />
                        )}

                        {result &&
                            ablationResults.map((ablationResult) => (
                                <AblationPredictions
                                    key={ablationResult.id}
                                    tokenLogits={ablationResult.tokenLogits}
                                    promptTokens={result.prompt_tokens}
                                    appliedMask={ablationResult.applied_mask}
                                    maskOverride={ablationResult.maskOverride}
                                />
                            ))}
                    </div>
                </div>

                <ComponentDetailModal
                    onClose={closePopup}
                    onToggleComponent={toggleComponentDisabled}
                    isComponentDisabled={isComponentDisabled}
                    promptId={currentPromptId}
                />
            </div>
        </main>
    );
}
