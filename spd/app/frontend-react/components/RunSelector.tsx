"use client";

import { useState } from "react";
import { api } from "@/lib/api";

interface RunSelectorProps {
    wandbRunId: string | null;
    setWandbRunId: (id: string) => void;
    loadingRun: boolean;
    setLoadingRun: (loading: boolean) => void;
    isLoading: boolean;
}

const presetRuns = [
    { id: "ry05f67a", label: "Run ry05f67a" },
    { id: "6a7en259", label: "Run 6a7en259" }
];

export default function RunSelector({
    wandbRunId,
    setWandbRunId,
    loadingRun,
    setLoadingRun,
    isLoading
}: RunSelectorProps) {
    const loadRun = async () => {
        if (!wandbRunId?.trim()) return;

        setLoadingRun(true);
        try {
            await api.loadRun(wandbRunId);
        } catch (error: any) {
            console.error(`Error loading run: ${error.message}`);
            alert(`Failed to load run: ${error.message}`);
        }
        setLoadingRun(false);
    };

    return (
        <div className="mb-4">
            <label htmlFor="wandb-run-id" className="block mb-2 font-semibold text-gray-800 text-sm">
                W&B Run ID
            </label>
            <div className="flex gap-2">
                <input
                    type="text"
                    id="wandb-run-id"
                    list="run-options"
                    value={wandbRunId || ""}
                    onChange={(e) => setWandbRunId(e.target.value)}
                    disabled={loadingRun || isLoading}
                    placeholder="Select or enter run ID"
                    className="flex-1 px-4 py-2 border border-gray-300 rounded text-base focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
                />
                <datalist id="run-options">
                    {presetRuns.map((preset) => (
                        <option key={preset.id} value={preset.id}>
                            {preset.label}
                        </option>
                    ))}
                </datalist>
                <button
                    onClick={loadRun}
                    disabled={loadingRun || isLoading || !wandbRunId?.trim()}
                    className={`px-4 py-2 rounded text-base whitespace-nowrap ${
                        loadingRun || isLoading || !wandbRunId?.trim()
                            ? "bg-gray-300 cursor-not-allowed"
                            : "bg-blue-500 text-white hover:bg-blue-600"
                    }`}
                >
                    {loadingRun ? "Loading..." : "Load Run"}
                </button>
            </div>
        </div>
    );
}