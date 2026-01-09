/**
 * API client for /api/correlations endpoints.
 */

import type { ComponentCorrelations, TokenStats } from "../localAttributionsTypes";
import { API_URL, fetchJson } from "./index";

export async function getComponentCorrelations(
    layer: string,
    componentIdx: number,
    topK: number,
): Promise<ComponentCorrelations | null> {
    const start = performance.now();
    const url = new URL(`${API_URL}/api/correlations/components/${encodeURIComponent(layer)}/${componentIdx}`);
    url.searchParams.set("top_k", String(topK));
    const result = await fetchJson<ComponentCorrelations | null>(url.toString());
    const elapsed = performance.now() - start;
    console.log(`[PERF] getComponentCorrelations(${layer}:${componentIdx}): ${elapsed.toFixed(1)}ms`);
    return result;
}

export async function getComponentTokenStats(
    layer: string,
    componentIdx: number,
    topK: number,
): Promise<TokenStats | null> {
    const start = performance.now();
    const url = new URL(`${API_URL}/api/correlations/token_stats/${encodeURIComponent(layer)}/${componentIdx}`);
    url.searchParams.set("top_k", String(topK));
    const result = await fetchJson<TokenStats | null>(url.toString());
    const elapsed = performance.now() - start;
    console.log(`[PERF] getComponentTokenStats(${layer}:${componentIdx}): ${elapsed.toFixed(1)}ms`);
    return result;
}

// Interpretation labels
export type Interpretation = {
    label: string;
    confidence: "low" | "medium" | "high";
    reasoning: string;
    prompt: string;
};

export async function getAllInterpretations(): Promise<Record<string, Interpretation>> {
    const start = performance.now();
    const result = await fetchJson<Record<string, Interpretation>>(`${API_URL}/api/correlations/interpretations`);
    const elapsed = performance.now() - start;
    console.log(`[PERF] getAllInterpretations: ${elapsed.toFixed(1)}ms (${Object.keys(result).length} interpretations)`);
    return result;
}

export async function getComponentInterpretation(layer: string, componentIdx: number): Promise<Interpretation | null> {
    const start = performance.now();
    const result = await fetchJson<Interpretation | null>(
        `${API_URL}/api/correlations/interpretations/${encodeURIComponent(layer)}/${componentIdx}`,
    );
    const elapsed = performance.now() - start;
    console.log(`[PERF] getComponentInterpretation(${layer}:${componentIdx}): ${elapsed.toFixed(1)}ms`);
    return result;
}

export async function requestComponentInterpretation(layer: string, componentIdx: number): Promise<Interpretation> {
    const start = performance.now();
    const result = await fetchJson<Interpretation>(
        `${API_URL}/api/correlations/interpretations/${encodeURIComponent(layer)}/${componentIdx}`,
        { method: "POST" },
    );
    const elapsed = performance.now() - start;
    console.log(`[PERF] requestComponentInterpretation(${layer}:${componentIdx}): ${elapsed.toFixed(1)}ms`);
    return result;
}
