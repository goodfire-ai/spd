/**
 * API client for /api/correlations endpoints.
 */

import type { ComponentCorrelations, TokenStats } from "../localAttributionsTypes";
import { API_URL, fetchJson } from "./index";

export async function getComponentCorrelations(
    layer: string,
    componentIdx: number,
    topK: number,
): Promise<ComponentCorrelations> {
    const url = new URL(`${API_URL}/api/correlations/components/${encodeURIComponent(layer)}/${componentIdx}`);
    url.searchParams.set("top_k", String(topK));
    return fetchJson<ComponentCorrelations>(url.toString());
}

export async function getComponentTokenStats(
    layer: string,
    componentIdx: number,
    topK: number,
): Promise<TokenStats | null> {
    const url = new URL(`${API_URL}/api/correlations/token_stats/${encodeURIComponent(layer)}/${componentIdx}`);
    url.searchParams.set("top_k", String(topK));
    return fetchJson<TokenStats | null>(url.toString());
}

// Interpretation headline (bulk-fetched) - lightweight data for badges
export type Interpretation = {
    label: string;
    confidence: "low" | "medium" | "high";
};

// Interpretation detail (fetched on-demand) - reasoning and prompt
export type InterpretationDetail = {
    reasoning: string;
    prompt: string;
};

export async function getAllInterpretations(): Promise<Record<string, Interpretation>> {
    return fetchJson<Record<string, Interpretation>>(`${API_URL}/api/correlations/interpretations`);
}

export async function getInterpretationDetail(layer: string, componentIdx: number): Promise<InterpretationDetail> {
    return fetchJson<InterpretationDetail>(
        `${API_URL}/api/correlations/interpretations/${encodeURIComponent(layer)}/${componentIdx}`,
    );
}

export async function requestComponentInterpretation(layer: string, componentIdx: number): Promise<Interpretation> {
    return fetchJson<Interpretation>(
        `${API_URL}/api/correlations/interpretations/${encodeURIComponent(layer)}/${componentIdx}`,
        { method: "POST" },
    );
}

// Global attributions types
export type AttributingComponent = {
    component_key: string;
    attribution: number;
};

export type GlobalAttributions = {
    top_positive_sources: AttributingComponent[];
    top_negative_sources: AttributingComponent[];
    top_positive_targets: AttributingComponent[];
    top_negative_targets: AttributingComponent[];
};

export async function getGlobalAttributions(
    layer: string,
    componentIdx: number,
    topK: number,
): Promise<GlobalAttributions | null> {
    const url = new URL(
        `${API_URL}/api/correlations/global_attributions/${encodeURIComponent(layer)}/${componentIdx}`,
    );
    url.searchParams.set("top_k", String(topK));
    return fetchJson<GlobalAttributions | null>(url.toString());
}
