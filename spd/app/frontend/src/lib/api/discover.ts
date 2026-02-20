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
};

export async function discoverRuns(): Promise<DiscoveredRun[]> {
    return fetchJson<DiscoveredRun[]>("/api/runs/discover");
}
