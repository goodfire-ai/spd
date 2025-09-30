"use client";

import type { OutputTokenLogit } from "@/lib/api";
import TokenPredictions from "./TokenPredictions";

interface OriginalPredictionsProps {
    tokenLogits: OutputTokenLogit[][];
    promptTokens: string[];
    title: string;
}

export default function OriginalPredictions({ tokenLogits, promptTokens, title }: OriginalPredictionsProps) {
    return (
        <div className="mb-6">
            <h3
                className="text-lg font-semibold mb-2 text-gray-800"
                dangerouslySetInnerHTML={{ __html: title }}
            />
            <TokenPredictions
                tokenLogits={tokenLogits}
                promptTokens={promptTokens}
                containerClass="original border-green-500"
            />
        </div>
    );
}