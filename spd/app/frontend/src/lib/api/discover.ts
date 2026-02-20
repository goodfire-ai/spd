/**
 * API client for /api/runs/discover endpoint.
 */

import { fetchJson } from "./index";

export type DiscoveredRun = {
    run_id: string;
    n_labels: number;
    has_harvest: boolean;
    has_detection: boolean;
    has_fuzzing: boolean;
    has_intruder: boolean;
    has_dataset_attributions: boolean;
    model_type: string | null;
    arch_summary: string | null;
    created_at: string | null;
};

export async function discoverRuns(): Promise<DiscoveredRun[]> {
    return fetchJson<DiscoveredRun[]>("/api/runs/discover");
}
