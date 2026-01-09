/**
 * API client for /api/activation_contexts endpoints.
 */

import type { ActivationContextsSummary, ComponentDetail, ComponentProbeResult } from "../localAttributionsTypes";
import { API_URL, fetchJson } from "./index";

// Types for activation contexts
export type SubcomponentActivationContexts = {
    subcomponent_idx: number;
    mean_ci: number;
    example_tokens: string[][];
    example_ci: number[][];
    example_component_acts: number[][];
};

export async function getActivationContextsSummary(): Promise<ActivationContextsSummary> {
    const start = performance.now();
    const result = await fetchJson<ActivationContextsSummary>(`${API_URL}/api/activation_contexts/summary`);
    const elapsed = performance.now() - start;
    const componentCount = Object.values(result).reduce((sum, arr) => sum + arr.length, 0);
    console.log(`[PERF] getActivationContextsSummary: ${elapsed.toFixed(1)}ms (${componentCount} components)`);
    return result;
}

export async function getComponentDetail(layer: string, componentIdx: number): Promise<ComponentDetail> {
    const start = performance.now();
    const result = await fetchJson<ComponentDetail>(
        `${API_URL}/api/activation_contexts/${encodeURIComponent(layer)}/${componentIdx}`,
    );
    const elapsed = performance.now() - start;
    console.log(`[PERF] getComponentDetail(${layer}:${componentIdx}): ${elapsed.toFixed(1)}ms`);
    return result;
}

export async function probeComponent(text: string, layer: string, componentIdx: number): Promise<ComponentProbeResult> {
    const start = performance.now();
    const result = await fetchJson<ComponentProbeResult>(`${API_URL}/api/activation_contexts/probe`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, layer, component_idx: componentIdx }),
    });
    const elapsed = performance.now() - start;
    console.log(`[PERF] probeComponent(${layer}:${componentIdx}): ${elapsed.toFixed(1)}ms`);
    return result;
}
