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

// Interpretation labels (prompt fetched separately via getInterpretationPrompt)
export type Interpretation = {
    label: string;
    confidence: "low" | "medium" | "high";
    reasoning: string;
};

export async function getAllInterpretations(): Promise<Record<string, Interpretation>> {
    return fetchJson<Record<string, Interpretation>>(`${API_URL}/api/correlations/interpretations`);
}

export async function getInterpretationPrompt(layer: string, componentIdx: number): Promise<string> {
    return fetchJson<string>(
        `${API_URL}/api/correlations/interpretations/${encodeURIComponent(layer)}/${componentIdx}/prompt`,
    );
}

export async function requestComponentInterpretation(layer: string, componentIdx: number): Promise<Interpretation> {
    return fetchJson<Interpretation>(
        `${API_URL}/api/correlations/interpretations/${encodeURIComponent(layer)}/${componentIdx}`,
        { method: "POST" },
    );
}
