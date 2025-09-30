"use client";

import { useState, useEffect } from "react";
import { useComponentState } from "@/lib/context/ComponentStateContext";
import { api } from "@/lib/api";
import type { CosineSimilarityData } from "@/lib/api";
import CosineSimilarityPlot from "./CosineSimilarityPlot";

interface ComponentDetailModalProps {
    onClose: () => void;
    onToggleComponent: (layerName: string, tokenIdx: number, componentIdx: number) => void;
    isComponentDisabled: (layerName: string, tokenIdx: number, componentIdx: number) => boolean;
    promptId: string | null;
}

export default function ComponentDetailModal({
    onClose,
    onToggleComponent,
    isComponentDisabled,
    promptId
}: ComponentDetailModalProps) {
    const { popupData, runAblation } = useComponentState();
    const [similarityData, setSimilarityData] = useState<CosineSimilarityData | null>(null);
    const [loadingSimilarities, setLoadingSimilarities] = useState(false);

    useEffect(() => {
        if (popupData && promptId) {
            loadCosineSimilarities(promptId, popupData.layer, popupData.tokenIdx);
        }
    }, [popupData, promptId]);

    const loadCosineSimilarities = async (promptId: string, layer: string, tokenIdx: number) => {
        setLoadingSimilarities(true);
        try {
            const data = await api.getCosineSimilarities(promptId, layer, tokenIdx);
            setSimilarityData(data);
        } catch (error) {
            console.error("Failed to load cosine similarities:", error);
            setSimilarityData(null);
        }
        setLoadingSimilarities(false);
    };

    if (!popupData) return null;

    const getColorFromCI = (ci: number): string => {
        const whiteAmount = Math.round((1 - ci) * 255);
        return `rgb(${whiteAmount}, ${whiteAmount}, 255)`;
    };

    const getAllComponentIndices = (): number[] => {
        return popupData.tokenCis.indices;
    };

    const areAllComponentsDisabled = (): boolean => {
        const allIndices = getAllComponentIndices();
        return allIndices.every((idx) => isComponentDisabled(popupData.layer, popupData.tokenIdx, idx));
    };

    const toggleAllComponents = () => {
        const allIndices = getAllComponentIndices();
        const shouldDisable = !areAllComponentsDisabled();

        for (const componentIdx of allIndices) {
            const isCurrentlyDisabled = isComponentDisabled(popupData.layer, popupData.tokenIdx, componentIdx);
            if (shouldDisable && !isCurrentlyDisabled) {
                onToggleComponent(popupData.layer, popupData.tokenIdx, componentIdx);
            } else if (!shouldDisable && isCurrentlyDisabled) {
                onToggleComponent(popupData.layer, popupData.tokenIdx, componentIdx);
            }
        }
    };

    const disabledComponentIndices = getAllComponentIndices().filter((idx) => {
        const layerAblations = runAblation[popupData.layer];
        if (!layerAblations || !layerAblations[popupData.tokenIdx]) return false;
        return layerAblations[popupData.tokenIdx].includes(idx);
    });

    return (
        <div
            className="fixed inset-0 bg-black/50 flex justify-center items-center z-[1000]"
            onClick={onClose}
        >
            <div
                className="bg-white rounded-lg p-0 max-w-[600px] max-h-[80vh] w-[90%] shadow-[0_4px_20px_rgba(0,0,0,0.3)] overflow-hidden"
                onClick={(e) => e.stopPropagation()}
            >
                <div className="p-6 overflow-y-auto max-h-[calc(80vh-80px)]">
                    <div className="mb-6 p-4 bg-gray-50 rounded">
                        <p className="my-2 text-gray-600">
                            <strong>Token:</strong> &quot;{popupData.token}&quot; (position {popupData.tokenIdx})
                        </p>
                        <p className="my-2 text-gray-600">
                            <strong>Layer:</strong> {popupData.layer}
                        </p>
                        <p className="my-2 text-gray-600">
                            <strong>L0 (Non-zero components):</strong> {popupData.tokenCis.l0}
                        </p>
                        <p className="my-2 text-gray-600">
                            <strong>Vector Length:</strong> {popupData.tokenCis.values.length}
                        </p>
                    </div>

                    <div>
                        <div className="flex justify-between items-center mb-4">
                            <h4 className="m-0 text-gray-800">Component Values:</h4>
                            <label className="flex items-center gap-2 text-sm text-gray-800 cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={areAllComponentsDisabled()}
                                    onChange={toggleAllComponents}
                                    className="cursor-pointer"
                                />
                                Select All
                            </label>
                        </div>
                        <div className="grid grid-cols-[repeat(auto-fill,minmax(120px,1fr))] gap-2 max-h-[300px] overflow-y-auto border border-gray-200 p-4 rounded">
                            {popupData.tokenCis.values.map((value, idx) => {
                                const componentIdx = popupData.tokenCis.indices[idx];
                                const disabled = isComponentDisabled(popupData.layer, popupData.tokenIdx, componentIdx);
                                return (
                                    <div
                                        key={idx}
                                        className={`flex justify-between items-center px-2 py-1 rounded-sm text-sm cursor-pointer transition-colors ${
                                            disabled ? "!bg-red-500 opacity-70" : ""
                                        }`}
                                        style={{ backgroundColor: disabled ? undefined : getColorFromCI(value / 3) }}
                                        onClick={() => onToggleComponent(popupData.layer, popupData.tokenIdx, componentIdx)}
                                    >
                                        <span className="text-gray-600 font-bold flex-shrink-0">{componentIdx}:</span>
                                        <span className="font-bold text-right flex-grow mx-2">{value.toFixed(4)}</span>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {loadingSimilarities ? (
                        <div className="mt-4 p-4 text-center text-gray-600 italic">
                            Loading similarity data...
                        </div>
                    ) : similarityData ? (
                        <div className="mt-6 pt-6 border-t border-gray-200">
                            <h3 className="m-0 mb-4 text-gray-800 text-base">Pairwise Cosine Similarities</h3>
                            <div className="flex gap-8 justify-around flex-wrap">
                                <CosineSimilarityPlot
                                    title="Input Singular Vectors"
                                    data={similarityData.input_singular_vectors}
                                    indices={similarityData.component_indices}
                                    disabledIndices={disabledComponentIndices}
                                />
                                <CosineSimilarityPlot
                                    title="Output Singular Vectors"
                                    data={similarityData.output_singular_vectors}
                                    indices={similarityData.component_indices}
                                    disabledIndices={disabledComponentIndices}
                                />
                            </div>
                        </div>
                    ) : null}
                </div>
            </div>
        </div>
    );
}