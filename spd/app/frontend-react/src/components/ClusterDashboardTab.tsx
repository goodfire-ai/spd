import React, { useState, useEffect, useRef } from 'react';
import { getClusterDashboardData } from '../api';
import type { ClusterDashboardResponse } from '../api';
import { ClusterDashboardBody } from './ClusterDashboardBody';

interface ClusterDashboardTabProps {
    iteration: number;
}

export const ClusterDashboardTab: React.FC<ClusterDashboardTabProps> = ({ iteration }) => {
    const [loading, setLoading] = useState(true);
    const [errorMsg, setErrorMsg] = useState<string | null>(null);
    const [dashboard, setDashboard] = useState<ClusterDashboardResponse | null>(null);

    const [nSamples, setNSamples] = useState(16);
    const [nBatches, setNBatches] = useState(2);
    const [batchSize, setBatchSize] = useState(64);
    const [contextLength, setContextLength] = useState(64);

    const pendingControllerRef = useRef<AbortController | null>(null);

    const fetchDashboard = async () => {
        pendingControllerRef.current?.abort();
        const controller = new AbortController();
        pendingControllerRef.current = controller;

        setLoading(true);
        setErrorMsg(null);

        try {
            console.log("fetching dashboard");
            const data = await getClusterDashboardData({
                iteration,
                n_samples: nSamples,
                n_batches: nBatches,
                batch_size: batchSize,
                context_length: contextLength,
            });
            setDashboard(data);
        } catch (e: any) {
            if (controller.signal.aborted) return;
            setErrorMsg(e?.message ?? String(e));
        } finally {
            if (!controller.signal.aborted) {
                setLoading(false);
            }
        }
    };

    useEffect(() => {
        fetchDashboard();
        return () => {
            pendingControllerRef.current?.abort();
            pendingControllerRef.current = null;
        };
    }, []);

    const refresh = (e: React.FormEvent) => {
        e.preventDefault();
        fetchDashboard().catch((e) => setErrorMsg(e?.message ?? String(e)));
    };

    return (
        <div className="tab-content">
            <div className="toolbar">
                <form className="toolbar-form" onSubmit={refresh}>
                    <label>
                        Samples
                        <input
                            type="number"
                            min={1}
                            value={nSamples}
                            onChange={(e) => setNSamples(parseInt(e.target.value))}
                        />
                    </label>
                    <label>
                        Batches
                        <input
                            type="number"
                            min={1}
                            value={nBatches}
                            onChange={(e) => setNBatches(parseInt(e.target.value))}
                        />
                    </label>
                    <label>
                        Batch Size
                        <input
                            type="number"
                            min={1}
                            value={batchSize}
                            onChange={(e) => setBatchSize(parseInt(e.target.value))}
                        />
                    </label>
                    <label>
                        Context
                        <input
                            type="number"
                            min={1}
                            value={contextLength}
                            onChange={(e) => setContextLength(parseInt(e.target.value))}
                        />
                    </label>
                    <button className="run-button" type="submit">
                        Run
                    </button>
                </form>
            </div>

            {loading ? (
                <div className="status">Loading...</div>
            ) : errorMsg ? (
                <div className="status-error">{errorMsg}</div>
            ) : dashboard ? (
                <ClusterDashboardBody dashboard={dashboard} />
            ) : null}
        </div>
    );
};
