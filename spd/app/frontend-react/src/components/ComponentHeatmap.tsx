import React from 'react';
import type { RunPromptResponse, MatrixCausalImportances } from '../api';

interface ComponentHeatmapProps {
    result: RunPromptResponse;
    promptId: string;
    onCellPopop: (
        token: string,
        tokenIdx: number,
        layerIdx: number,
        layerName: string,
        tokenCIs: MatrixCausalImportances
    ) => void;
    onMaskCreated?: () => void;
}

export const ComponentHeatmap: React.FC<ComponentHeatmapProps> = ({
    result,
    promptId,
    onCellPopop,
    onMaskCreated,
}) => {
    return (
        <div className="component-heatmap">
            <h3>Component Heatmap</h3>
            <p>TODO: Implement heatmap visualization</p>
        </div>
    );
};
