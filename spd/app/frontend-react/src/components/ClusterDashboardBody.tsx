import React from 'react';
import type { ClusterDashboardResponse } from '../api';

interface ClusterDashboardBodyProps {
    dashboard: ClusterDashboardResponse;
}

export const ClusterDashboardBody: React.FC<ClusterDashboardBodyProps> = ({ dashboard }) => {
    return (
        <div className="cluster-dashboard-body">
            <h3>Cluster Dashboard</h3>
            <p>TODO: Implement cluster dashboard visualization</p>
            <p>Clusters: {dashboard.clusters.length}</p>
            <p>Samples: {dashboard.text_samples.length}</p>
        </div>
    );
};
