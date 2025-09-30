"use client";

import type { OutputTokenLogit, ComponentMask, MaskOverrideDTO } from "@/lib/api";
import TokenPredictions from "./TokenPredictions";

interface AblationPredictionsProps {
    tokenLogits: OutputTokenLogit[][];
    promptTokens: string[];
    appliedMask: ComponentMask;
    maskOverride?: MaskOverrideDTO;
}

export default function AblationPredictions({
    tokenLogits,
    promptTokens,
    appliedMask,
    maskOverride
}: AblationPredictionsProps) {
    return (
        <div className="mb-6">
            <h3 className="text-lg font-semibold mb-2 text-orange-600">
                Ablation Results
                {maskOverride && (
                    <span className="ml-2 text-sm font-normal text-gray-600">
                        (Mask: {maskOverride.description || "Unnamed"} - Layer: {maskOverride.layer})
                    </span>
                )}
            </h3>
            <TokenPredictions
                tokenLogits={tokenLogits}
                promptTokens={promptTokens}
                appliedMask={appliedMask}
            />
        </div>
    );
}