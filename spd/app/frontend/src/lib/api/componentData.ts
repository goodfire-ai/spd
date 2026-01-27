/**
 * API client for /api/component_data bulk endpoint.
 *
 * Fetches all component data in a single request to avoid GIL contention
 * and reduce HTTP roundtrip overhead.
 */

import type {
    ComponentCorrelationsResponse,
    SubcomponentActivationContexts,
    TokenStatsResponse,
} from "../promptAttributionsTypes";
import { API_URL, fetchJson } from "./index";

export interface BulkComponentDataResponse {
    activation_contexts: Record<string, SubcomponentActivationContexts>;
    correlations: Record<string, ComponentCorrelationsResponse>;
    token_stats: Record<string, TokenStatsResponse>;
}

/**
 * Bulk fetch all component data in a single request.
 *
 * Combines activation contexts, correlations, and token stats.
 * This eliminates GIL contention from multiple concurrent requests
 * and reduces HTTP roundtrips from 3 to 1.
 */
export async function getComponentDataBulk(
    componentKeys: string[],
    activationContextsLimit: number = 30,
    correlationsTopK: number = 20,
    tokenStatsTopK: number = 30,
): Promise<BulkComponentDataResponse> {
    return fetchJson<BulkComponentDataResponse>(`${API_URL}/api/component_data/bulk`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            component_keys: componentKeys,
            activation_contexts_limit: activationContextsLimit,
            correlations_top_k: correlationsTopK,
            token_stats_top_k: tokenStatsTopK,
        }),
    });
}
