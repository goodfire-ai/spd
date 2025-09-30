"use client";

import { useComponentState } from "@/lib/context/ComponentStateContext";
import type { LayerCIs, SparseVector, RunPromptResponse } from "@/lib/api";
import MaskCombinePanel from "./MaskCombinePanel";

interface ComponentHeatmapProps {
    result: RunPromptResponse;
    promptId: string;
    onCellClick: (
        token: string,
        tokenIdx: number,
        layer: string,
        layerIdx: number,
        tokenCis: SparseVector
    ) => void;
    onMaskCreated: () => void;
}

export default function ComponentHeatmap({ result, promptId, onCellClick, onMaskCreated }: ComponentHeatmapProps) {
    const { runAblation, multiSelectMode, selectedTokensForCombining, setSelectedTokensForCombining } = useComponentState();

    const globalMax = Math.max(
        ...result.layer_cis.flatMap((layer) => layer.token_cis.map((tokenCIs) => tokenCIs.l0))
    );

    const layer_cis = [...result.layer_cis].reverse();

    const handleCellClick = (
        token: string,
        tokenIdx: number,
        layer: string,
        layerIdx: number,
        tokenCis: SparseVector
    ) => {
        if (multiSelectMode) {
            const existingIndex = selectedTokensForCombining.findIndex(
                (t) => t.layer === layer && t.tokenIdx === tokenIdx
            );

            if (existingIndex >= 0) {
                setSelectedTokensForCombining(
                    selectedTokensForCombining.filter((_, idx) => idx !== existingIndex)
                );
            } else {
                setSelectedTokensForCombining([
                    ...selectedTokensForCombining,
                    { layer, tokenIdx, token }
                ]);
            }
        } else {
            onCellClick(token, tokenIdx, layer, layerIdx, tokenCis);
        }
    };

    const isTokenSelected = (layer: string, tokenIdx: number): boolean => {
        return selectedTokensForCombining.some((t) => t.layer === layer && t.tokenIdx === tokenIdx);
    };

    const getColorFromL0 = (l0: number, layerName: string, tokenIdx: number): string => {
        const intensity = Math.max(0, Math.min(1, l0 / globalMax));
        const disabledComponents = runAblation[layerName]?.[tokenIdx]?.length ?? 0;
        const totalComponents = l0;
        const disabledRatio = totalComponents > 0 ? disabledComponents / totalComponents : 0;

        const whiteAmount = Math.round((1 - intensity) * 255);
        const baseColor = `rgb(${whiteAmount}, ${whiteAmount}, 255)`;

        if (disabledRatio === 0) {
            return baseColor;
        }

        const disabledPercent = Math.round(disabledRatio * 100);
        return `linear-gradient(to right, #ff4444 0%, #ff4444 ${disabledPercent}%, ${baseColor} ${disabledPercent}%, ${baseColor} 100%)`;
    };

    return (
        <div className="border border-gray-300 rounded p-4 bg-gray-50">
            <MaskCombinePanel promptId={promptId} onMaskCreated={onMaskCreated} />
            <div className="flex">
                <div className="flex flex-col mr-2 flex-shrink-0">
                    <div className="h-10 order-[999]"></div>
                    {layer_cis.map((layer) => (
                        <div
                            key={layer.module}
                            className="h-5 flex items-center justify-end pr-2 text-sm font-bold text-gray-600 min-w-[100px] mb-0.5"
                        >
                            {layer.module}
                        </div>
                    ))}
                </div>

                <div className="flex-1 overflow-x-auto">
                    <div className="flex flex-col min-w-fit">
                        {layer_cis.map((layer, layerIdx) => (
                            <div key={layer.module} className="flex mb-0.5">
                                {result.prompt_tokens.map((token, tokenIdx) => (
                                    <div
                                        key={tokenIdx}
                                        className={`w-[50px] h-5 border border-white cursor-pointer transition-transform ${
                                            isTokenSelected(layer.module, tokenIdx)
                                                ? "!border-[3px] !border-green-500 shadow-[0_0_8px_rgba(76,175,80,0.7)] relative z-[5]"
                                                : ""
                                        } ${multiSelectMode ? "hover:border-2 hover:border-green-500 hover:shadow-[0_0_5px_rgba(76,175,80,0.5)]" : "hover:border-2 hover:border-[#241d8c] hover:z-10 hover:relative"}`}
                                        style={{ background: getColorFromL0(layer.token_cis[tokenIdx].l0, layer.module, tokenIdx) }}
                                        title={`L0=${layer.token_cis[tokenIdx].l0}`}
                                        onClick={() =>
                                            handleCellClick(token, tokenIdx, layer.module, layerIdx, layer.token_cis[tokenIdx])
                                        }
                                    >
                                        {isTokenSelected(layer.module, tokenIdx) && (
                                            <span className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-white text-sm font-bold [text-shadow:0_0_3px_rgba(0,0,0,0.7)]">
                                                ✓
                                            </span>
                                        )}
                                    </div>
                                ))}
                            </div>
                        ))}

                        <div className="flex h-10 mb-0.5">
                            {result.prompt_tokens.map((token, idx) => (
                                <div
                                    key={idx}
                                    className="w-[50px] flex items-center justify-center text-xs font-bold text-gray-800 text-center px-0.5 break-all border-r border-gray-200"
                                >
                                    {token}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}