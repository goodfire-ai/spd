"use client";

import { useState, useEffect, forwardRef, useImperativeHandle } from "react";
import { api } from "@/lib/api";
import type { MaskOverrideDTO } from "@/lib/api";

interface SavedMasksPanelProps {
    onApplyMask: (maskId: string) => void;
}

export interface SavedMasksPanelRef {
    loadMasks: () => Promise<void>;
}

const SavedMasksPanel = forwardRef<SavedMasksPanelRef, SavedMasksPanelProps>(
    ({ onApplyMask }, ref) => {
        const [savedMasks, setSavedMasks] = useState<MaskOverrideDTO[]>([]);
        const [loading, setLoading] = useState(false);

        const loadMasks = async () => {
            setLoading(true);
            try {
                const masks = await api.getMaskOverrides();
                setSavedMasks(masks);
            } catch (error) {
                console.error("Failed to load mask overrides:", error);
            } finally {
                setLoading(false);
            }
        };

        useImperativeHandle(ref, () => ({
            loadMasks
        }));

        useEffect(() => {
            loadMasks();
        }, []);

        return (
            <div className="mb-4 p-4 bg-gray-50 border border-gray-300 rounded-lg">
                <div className="flex justify-between items-center mb-4">
                    <h3 className="m-0 text-lg text-gray-800">Saved Masks</h3>
                    <button
                        className={`px-3 py-1 text-white rounded text-sm ${
                            loading ? "bg-gray-400 cursor-not-allowed" : "bg-blue-500 hover:bg-blue-600"
                        }`}
                        onClick={loadMasks}
                        disabled={loading}
                    >
                        {loading ? "Loading..." : "Refresh"}
                    </button>
                </div>

                {savedMasks.length === 0 ? (
                    <div className="py-8 text-center text-gray-500 italic">
                        No saved masks yet. Create masks using multi-select mode below.
                    </div>
                ) : (
                    <div className="flex gap-3 overflow-x-auto pb-2">
                        {savedMasks.map((mask) => (
                            <div
                                key={mask.id}
                                className="flex-[0_0_auto] min-w-[200px] p-3 bg-white border-2 border-gray-200 rounded-md transition-all hover:border-blue-500 hover:shadow-md"
                            >
                                <div className="mb-3">
                                    <div className="font-medium text-gray-800 mb-2 overflow-hidden text-ellipsis whitespace-nowrap">
                                        {mask.description || "Unnamed mask"}
                                    </div>
                                    <div className="flex flex-col gap-1 text-sm text-gray-600">
                                        <span className="font-medium">Layer: {mask.layer}</span>
                                        <span className="text-blue-500">L0: {mask.combined_mask.l0}</span>
                                    </div>
                                </div>
                                <button
                                    className="w-full px-4 py-2 bg-blue-500 text-white rounded font-medium transition-colors hover:bg-blue-600"
                                    onClick={() => onApplyMask(mask.id)}
                                >
                                    Apply as Ablation
                                </button>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        );
    }
);

SavedMasksPanel.displayName = "SavedMasksPanel";

export default SavedMasksPanel;