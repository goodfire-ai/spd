import React from 'react';
import type { Status } from '../api';

interface SidebarProps {
    status: Status | null;
    trainWandbRunId: string;
    setTrainWandbRunId: (id: string) => void;
    loadingTrainRun: boolean;
    onLoadRun: () => void;
    activeTab: string | null;
    setActiveTab: (tab: string | null) => void;
    clusterWandbRunPath: string | null;
    setClusterWandbRunPath: (path: string | null) => void;
    clusterIteration: number | null;
    setClusterIteration: (iteration: number | null) => void;
    loadingClusterRun: boolean;
    onLoadClusterRun: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
    status,
    trainWandbRunId,
    setTrainWandbRunId,
    loadingTrainRun,
    onLoadRun,
    activeTab,
    setActiveTab,
    clusterWandbRunPath,
    setClusterWandbRunPath,
    clusterIteration,
    setClusterIteration,
    loadingClusterRun,
    onLoadClusterRun,
}) => {
};
