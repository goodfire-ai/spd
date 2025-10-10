import React from 'react';
import type { OutputTokenLogit } from '../api';

interface OriginalPredictionsProps {
    tokenLogits: OutputTokenLogit[][];
    promptTokens: string[];
    title: string;
}

export const OriginalPredictions: React.FC<OriginalPredictionsProps> = ({
    tokenLogits,
    promptTokens,
    title,
}) => {
    return (
        <div className="original-predictions">
            <h3 dangerouslySetInnerHTML={{ __html: title }}></h3>
            <p>TODO: Show predictions</p>
        </div>
    );
};
