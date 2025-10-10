import React from 'react';

interface SavedMasksPanelProps {
    onApplyMask: (maskId: string) => void;
}

export const SavedMasksPanel: React.FC<SavedMasksPanelProps> = ({ onApplyMask }) => {
    return (
        <div className="saved-masks-panel">
            <h3>Saved Masks</h3>
            <p>TODO: Show saved masks</p>
        </div>
    );
};
