import React from 'react';

interface DisabledComponentsPanelProps {
    promptTokens: string[];
    isLoading: boolean;
    onSendAblation: () => void;
    onToggleComponent: (layerName: string, tokenIdx: number, componentIdx: number) => void;
}

export const DisabledComponentsPanel: React.FC<DisabledComponentsPanelProps> = ({
    promptTokens,
    isLoading,
    onSendAblation,
    onToggleComponent,
}) => {
    return (
        <div className="disabled-components-panel">
            <h3>Disabled Components</h3>
            <button onClick={onSendAblation} disabled={isLoading}>
                Send Ablation
            </button>
            <p>TODO: Show disabled components</p>
        </div>
    );
};
