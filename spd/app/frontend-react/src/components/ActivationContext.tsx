import React from 'react';
import type { ActivationContext as ActivationContextType } from '../api';
import { TokenHighlights } from './TokenHighlights';

interface ActivationContextProps {
    example: ActivationContextType;
}

export const ActivationContext: React.FC<ActivationContextProps> = ({ example }) => {
    return (
        <div className="example-item">
            <TokenHighlights
                rawText={example.raw_text}
                offsetMapping={example.offset_mapping}
                tokenCiValues={example.token_ci_values}
                activePosition={example.active_position}
            />
        </div>
    );
};
