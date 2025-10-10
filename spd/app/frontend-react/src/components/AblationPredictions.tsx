import React from 'react';
import type { OutputTokenLogit, ComponentMask, MaskOverrideDTO, AblationStats } from '../api';

interface AblationPredictionsProps {
    tokenLogits: OutputTokenLogit[][];
    promptTokens: string[];
    appliedMask: ComponentMask;
    maskOverride?: MaskOverrideDTO;
    ablationStats: AblationStats;
}

export const AblationPredictions: React.FC<AblationPredictionsProps> = ({
    tokenLogits,
    promptTokens,
    appliedMask,
    maskOverride,
    ablationStats,
}) => {
    return (
        <div className="ablation-predictions">
            <h3>Ablation Predictions</h3>
            <p>TODO: Show ablation predictions</p>
        </div>
    );
};
