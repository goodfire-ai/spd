/**
 * API client for /api/component_data bundled endpoint.
 *
 * Fetches all ComponentNodeCard data in a single request to reduce
 * HTTP roundtrip overhead (significant when using SSH port forwarding).
 */

import type { ComponentCorrelationsResponse, SubcomponentActivationContexts, TokenStatsResponse } from "../promptAttributionsTypes";
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
