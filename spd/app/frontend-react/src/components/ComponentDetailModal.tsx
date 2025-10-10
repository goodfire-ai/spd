import React from 'react';
import type { ClusterRunDTO, MatrixCausalImportances, ClusterDashboardResponse } from '../api';

interface PopupData {
    token: string;
    tokenIdx: number;
    layerIdx: number;
    layerName: string;
    tokenCIs: MatrixCausalImportances;
}

interface ComponentDetailModalProps {
    cluster: ClusterRunDTO;
    popupData: PopupData;
    dashboard: ClusterDashboardResponse;
    onClose: () => void;
    toggleComponent: (layerName: string, tokenIdx: number, componentIdx: number) => void;
    isComponentDisabled: (layerName: string, tokenIdx: number, componentIdx: number) => boolean;
}

export const ComponentDetailModal: React.FC<ComponentDetailModalProps> = ({
    cluster,
    popupData,
    dashboard,
    onClose,
    toggleComponent,
    isComponentDisabled,
}) => {
    return (
        <div className="component-detail-modal-overlay" onClick={onClose}>
            <div className="component-detail-modal" onClick={(e) => e.stopPropagation()}>
                <button className="close-button" onClick={onClose}>
                    ×
                </button>
                <h3>Component Details</h3>
                <p>TODO: Show component details</p>
            </div>
        </div>
    );
};
