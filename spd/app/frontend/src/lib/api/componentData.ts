/** API client for bulk component data endpoint. */

import type {
    SubcomponentCorrelationsResponse,
    SubcomponentActivationContexts,
    TokenStatsResponse,
} from "../promptAttributionsTypes";
import { fetchJson } from "./index";

export interface BulkComponentDataResponse {
    activation_contexts: Record<string, SubcomponentActivationContexts>;
    correlations: Record<string, SubcomponentCorrelationsResponse>;
    token_stats: Record<string, TokenStatsResponse>;
}

/**
 * Bulk fetch all component data in a single request.
 * Combines activation contexts, correlations, and token stats.
 */
export async function getComponentDataBulk(
    componentKeys: string[],
    activationContextsLimit: number,
    correlationsTopK: number,
    tokenStatsTopK: number,
): Promise<BulkComponentDataResponse> {
    return fetchJson<BulkComponentDataResponse>("/api/component_data/bulk", {
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
