"use client";

import { useComponentState } from "@/lib/context/ComponentStateContext";

interface DisabledComponentsPanelProps {
    promptTokens: string[];
    isLoading: boolean;
    onSendAblation: () => void;
    onToggleComponent: (layerName: string, tokenIdx: number, componentIdx: number) => void;
}

export default function DisabledComponentsPanel({
    promptTokens,
    isLoading,
    onSendAblation,
    onToggleComponent
}: DisabledComponentsPanelProps) {
    const { runAblation } = useComponentState();

    const hasDisabledComponents = Object.keys(runAblation).some((layer) =>
        runAblation[layer].some((tokenList) => tokenList.length > 0)
    );

    return (
        <div className="flex-[0_0_300px] p-4 border border-gray-300 rounded bg-gray-50 mt-4">
            <div className="flex justify-between items-center mb-4">
                <h3 className="m-0 text-gray-800 text-lg">Disabled Components</h3>
                <button
                    onClick={onSendAblation}
                    disabled={isLoading}
                    className={`px-4 py-2 text-white rounded text-sm ${
                        isLoading
                            ? "bg-gray-400 cursor-not-allowed"
                            : "bg-orange-600 hover:bg-orange-700"
                    }`}
                >
                    {isLoading ? "Sending..." : "Run with ablations"}
                </button>
            </div>
            {hasDisabledComponents ? (
                <div className="flex flex-col gap-4">
                    {Object.entries(runAblation).map(([layerName, tokenArrays]) =>
                        tokenArrays.map((disabledComponents, tokenIdx) =>
                            disabledComponents.length > 0 ? (
                                <div
                                    key={`${layerName}-${tokenIdx}`}
                                    className="bg-white p-3 rounded border border-gray-300"
                                >
                                    <div className="mb-2 text-sm text-gray-600">
                                        <strong>{promptTokens[tokenIdx]}</strong> in <em>{layerName}</em>
                                    </div>
                                    <div className="flex flex-wrap gap-1">
                                        {disabledComponents.map((componentIdx) => (
                                            <span
                                                key={componentIdx}
                                                className="bg-red-500 text-white px-2 py-1 rounded-xl text-xs cursor-pointer hover:bg-red-600 transition-colors"
                                                onClick={() => onToggleComponent(layerName, tokenIdx, componentIdx)}
                                            >
                                                {componentIdx} ×
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            ) : null
                        )
                    )}
                </div>
            ) : (
                <p className="text-gray-500 italic m-0 text-center">No components disabled yet</p>
            )}
        </div>
    );
}