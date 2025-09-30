"use client";

import { useRef, useEffect } from "react";
import type { OutputTokenLogit, ComponentMask } from "@/lib/api";

interface TokenPredictionsProps {
    tokenLogits: OutputTokenLogit[][];
    promptTokens: string[];
    containerClass?: string;
    appliedMask?: ComponentMask | null;
}

let isScrolling = false;

export default function TokenPredictions({
    tokenLogits,
    promptTokens,
    containerClass = "",
    appliedMask = null
}: TokenPredictionsProps) {
    const containerRef = useRef<HTMLDivElement>(null);

    const hasAblation = (tokenIdx: number): boolean => {
        if (!appliedMask) return false;
        return Object.values(appliedMask).some(
            (tokenMasks) => tokenMasks[tokenIdx] && tokenMasks[tokenIdx].length > 0
        );
    };

    const getProbabilityColor = (probability: number): string => {
        const p = Math.max(0, Math.min(1, probability));
        const whiteAmount = Math.round(255 - (255 - 100) * p);
        return `rgb(${whiteAmount}, ${whiteAmount}, 255)`;
    };

    const syncScroll = (event: Event) => {
        if (isScrolling) return;

        const target = event.target as HTMLElement;
        const scrollLeft = target.scrollLeft;

        isScrolling = true;

        const containers = document.querySelectorAll(".logits-display-container");
        for (const container of containers) {
            if (container !== target) {
                (container as HTMLElement).scrollLeft = scrollLeft;
            }
        }

        setTimeout(() => {
            isScrolling = false;
        }, 10);
    };

    useEffect(() => {
        const container = containerRef.current;
        if (container) {
            container.addEventListener("scroll", syncScroll);
            return () => container.removeEventListener("scroll", syncScroll);
        }
    }, []);

    return (
        <div
            ref={containerRef}
            className={`logits-display-container overflow-x-auto border border-gray-300 rounded-md bg-white ${containerClass} [scrollbar-width:none] [&::-webkit-scrollbar]:hidden`}
        >
            <div className="flex flex-row flex-nowrap gap-0 min-w-fit p-2 w-max overflow-visible">
                {tokenLogits.map((tokenPredictions, tokenIdx) => (
                    <div
                        key={tokenIdx}
                        className={`bg-white rounded-md border w-[140px] p-1 mr-1 ${
                            hasAblation(tokenIdx)
                                ? "border-2 border-orange-500 bg-orange-50"
                                : "border-gray-300"
                        }`}
                    >
                        <div className="text-gray-800 text-center relative">
                            <div className="text-xs font-mono mt-1 break-all">
                                &quot;{promptTokens[tokenIdx]}&quot;
                            </div>
                        </div>
                        <div className="flex flex-col gap-0.5">
                            {tokenPredictions.map((prediction, idx) => (
                                <div
                                    key={idx}
                                    className="flex flex-row justify-between items-center rounded-sm text-xs px-1 py-0.5 my-0.5"
                                    style={{ backgroundColor: getProbabilityColor(prediction.probability) }}
                                >
                                    <span className="font-mono text-xs text-gray-800 overflow-hidden whitespace-nowrap text-ellipsis text-left flex-1 mr-1 px-1 py-0.5 rounded-sm">
                                        &quot;{prediction.token}&quot;
                                    </span>
                                    <span className="font-mono text-[0.65rem] text-right flex-shrink-0">
                                        {prediction.probability.toFixed(3)}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}