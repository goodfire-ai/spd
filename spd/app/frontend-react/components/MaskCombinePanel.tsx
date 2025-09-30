"use client";

import { useState, useEffect } from "react";
import { useComponentState } from "@/lib/context/ComponentStateContext";
import { api } from "@/lib/api";

interface MaskCombinePanelProps {
    promptId: string;
    onMaskCreated: () => void;
}

export default function MaskCombinePanel({ promptId, onMaskCreated }: MaskCombinePanelProps) {
    const { multiSelectMode, setMultiSelectMode, selectedTokensForCombining, setSelectedTokensForCombining } = useComponentState();
    const [combining, setCombining] = useState(false);
    const [description, setDescription] = useState("");
    const [simulatedL0, setSimulatedL0] = useState<number | null>(null);
    const [simulatedJacc, setSimulatedJacc] = useState<number | null>(null);
    const [simulating, setSimulating] = useState(false);

    const layersWithSelections = Array.from(
        new Set(selectedTokensForCombining.map((token) => token.layer))
    );

    useEffect(() => {
        if (selectedTokensForCombining.length > 0 && layersWithSelections.length === 1) {
            simulateMergeL0();
        } else {
            setSimulatedL0(null);
            setSimulatedJacc(null);
        }
    }, [selectedTokensForCombining]);

    const toggleMultiSelectMode = () => {
        setMultiSelectMode(!multiSelectMode);
        if (multiSelectMode) {
            setSelectedTokensForCombining([]);
            setSimulatedL0(null);
            setSimulatedJacc(null);
        }
    };

    const clearSelections = () => {
        setSelectedTokensForCombining([]);
        setSimulatedL0(null);
        setSimulatedJacc(null);
    };

    const simulateMergeL0 = async () => {
        if (layersWithSelections.length !== 1) {
            setSimulatedL0(null);
            setSimulatedJacc(null);
            return;
        }

        const layer = layersWithSelections[0];
        const tokenIndices = selectedTokensForCombining
            .filter((t) => t.layer === layer)
            .map((t) => t.tokenIdx);

        if (tokenIndices.length === 0) {
            setSimulatedL0(null);
            setSimulatedJacc(null);
            return;
        }

        setSimulating(true);
        try {
            const response = await api.simulateMerge(promptId, layer, tokenIndices);
            setSimulatedL0(response.l0);
            setSimulatedJacc(response.jacc);
        } catch (error) {
            console.error("Failed to simulate merge:", error);
            setSimulatedL0(null);
            setSimulatedJacc(null);
        } finally {
            setSimulating(false);
        }
    };

    const combineMasks = async () => {
        if (combining || layersWithSelections.length !== 1) return;

        const layer = layersWithSelections[0];
        const tokenIndices = selectedTokensForCombining
            .filter((t) => t.layer === layer)
            .map((t) => t.tokenIdx);

        setCombining(true);
        try {
            await api.combineMasks(promptId, layer, tokenIndices, description);
            setSelectedTokensForCombining([]);
            setMultiSelectMode(false);
            setDescription("");
            setSimulatedL0(null);
            setSimulatedJacc(null);
            onMaskCreated();
        } catch (error) {
            console.error("Failed to combine masks:", error);
        } finally {
            setCombining(false);
        }
    };

    return (
        <div className="p-3 border border-gray-300 rounded-lg mb-4 bg-white">
            <div className="flex gap-2 items-center flex-wrap">
                <button
                    className={`px-4 py-2 border-2 rounded font-bold transition-all ${
                        multiSelectMode
                            ? "bg-blue-500 text-white border-blue-500"
                            : "bg-white text-blue-500 border-blue-500 hover:bg-blue-50"
                    }`}
                    onClick={toggleMultiSelectMode}
                >
                    {multiSelectMode ? "✓ Multi-Select Mode" : "Enable Multi-Select"}
                </button>
                <input
                    type="text"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    placeholder="Description"
                    className="px-3 py-2 border border-gray-300 rounded"
                />

                {multiSelectMode && selectedTokensForCombining.length > 0 && (
                    <>
                        <button
                            className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
                            onClick={clearSelections}
                        >
                            Clear ({selectedTokensForCombining.length})
                        </button>

                        {simulatedL0 !== null && (
                            <span className="px-4 py-2 bg-blue-100 border border-blue-500 rounded text-blue-700 font-bold text-sm">
                                L0: {simulating ? "..." : simulatedL0}
                            </span>
                        )}

                        {simulatedJacc !== null && (
                            <span className="px-4 py-2 bg-blue-100 border border-blue-500 rounded text-blue-700 font-bold text-sm">
                                Jacc: {simulating ? "..." : simulatedJacc.toFixed(3)}
                            </span>
                        )}

                        <button
                            className={`px-4 py-2 rounded font-bold ${
                                layersWithSelections.length !== 1 || combining
                                    ? "bg-gray-300 cursor-not-allowed"
                                    : "bg-green-500 text-white hover:bg-green-600"
                            }`}
                            disabled={layersWithSelections.length !== 1 || combining}
                            onClick={combineMasks}
                        >
                            {combining ? "Combining..." : "Combine Masks"}
                        </button>

                        {layersWithSelections.length > 1 && (
                            <div className="flex items-start gap-2 p-3 bg-yellow-50 border border-yellow-200 border-l-4 border-l-orange-500 rounded-md text-yellow-900 max-w-md">
                                <span className="text-xl mt-0.5">⚠️</span>
                                <span className="text-sm leading-relaxed font-medium">
                                    Multi-select only available on a single layer. Please clear selections and select from one layer only.
                                </span>
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
}