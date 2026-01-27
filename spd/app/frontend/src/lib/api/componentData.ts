/**
 * API client for /api/component_data bundled endpoint.
 *
 * Fetches all ComponentNodeCard data in a single request to reduce
 * HTTP roundtrip overhead (significant when using SSH port forwarding).
 */

import type {
    ComponentCorrelationsResponse,
    SubcomponentActivationContexts,
    TokenStatsResponse,
} from "../promptAttributionsTypes";
import type { InterpretationDetail } from "./correlations";
import type { ComponentAttributions } from "./datasetAttributions";
import { API_URL, fetchJson } from "./index";

export interface ComponentDataBundle {
    component_detail: SubcomponentActivationContexts | null;
    correlations: ComponentCorrelationsResponse | null;
    token_stats: TokenStatsResponse | null;
    attributions: ComponentAttributions | null;
    interpretation_detail: InterpretationDetail | null;
    errors: Record<string, string>;
}

/**
 * Fetch all component data in a single bundled request.
 *
 * Eliminates 4 HTTP roundtrips vs the 5 individual endpoints,
 * saving ~400ms+ over SSH tunnels.
 */
export async function getComponentDataBundle(
    layer: string,
    componentIdx: number,
    topKCorr: number = 100,
    topKTokens: number = 50,
    topKAttrs: number = 20,
    detailLimit: number = 30,
): Promise<ComponentDataBundle> {
    const params = new URLSearchParams({
        top_k_corr: String(topKCorr),
        top_k_tokens: String(topKTokens),
        top_k_attrs: String(topKAttrs),
        detail_limit: String(detailLimit),
    });
    return fetchJson<ComponentDataBundle>(
        `${API_URL}/api/component_data/${encodeURIComponent(layer)}/${componentIdx}?${params}`,
    );
}

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
