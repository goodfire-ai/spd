/**
 * API client for /api/correlations endpoints.
 */

import type { ComponentCorrelationsResponse, TokenStatsResponse } from "../promptAttributionsTypes";
import { API_URL, fetchJson } from "./index";

export async function getComponentCorrelations(
    layer: string,
    componentIdx: number,
    topK: number,
): Promise<ComponentCorrelationsResponse> {
    const url = new URL(`${API_URL}/api/correlations/components/${encodeURIComponent(layer)}/${componentIdx}`);
    url.searchParams.set("top_k", String(topK));
    return fetchJson<ComponentCorrelationsResponse>(url.toString());
}

export async function getComponentTokenStats(
    layer: string,
    componentIdx: number,
    topK: number,
): Promise<TokenStatsResponse | null> {
    const url = new URL(`${API_URL}/api/correlations/token_stats/${encodeURIComponent(layer)}/${componentIdx}`);
    url.searchParams.set("top_k", String(topK));
    return fetchJson<TokenStatsResponse | null>(url.toString());
}

// Interpretation headline (bulk-fetched) - lightweight data for badges
export type InterpretationHeadline = {
    label: string;
    confidence: "low" | "medium" | "high";
};

// Interpretation detail (fetched on-demand) - reasoning and prompt
export type InterpretationDetail = {
    reasoning: string;
    prompt: string;
};

export async function getAllInterpretations(): Promise<Record<string, InterpretationHeadline>> {
    return fetchJson<Record<string, InterpretationHeadline>>(`${API_URL}/api/correlations/interpretations`);
}

export async function getInterpretationDetail(layer: string, componentIdx: number): Promise<InterpretationDetail> {
    return fetchJson<InterpretationDetail>(
        `${API_URL}/api/correlations/interpretations/${encodeURIComponent(layer)}/${componentIdx}`,
    );
}

export async function requestComponentInterpretation(
    layer: string,
    componentIdx: number,
): Promise<InterpretationHeadline> {
    return fetchJson<InterpretationHeadline>(
        `${API_URL}/api/correlations/interpretations/${encodeURIComponent(layer)}/${componentIdx}`,
        { method: "POST" },
    );
}
