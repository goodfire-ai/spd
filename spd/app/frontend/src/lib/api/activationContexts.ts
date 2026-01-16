/**
 * API client for /api/activation_contexts endpoints.
 */

import type { ActivationContextsSummary, ComponentDetail, ComponentProbeResult } from "../promptAttributionsTypes";
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
    return fetchJson<ActivationContextsSummary>(`${API_URL}/api/activation_contexts/summary`);
}

export async function getComponentDetail(layer: string, componentIdx: number): Promise<ComponentDetail> {
    return fetchJson<ComponentDetail>(
        `${API_URL}/api/activation_contexts/${encodeURIComponent(layer)}/${componentIdx}`,
    );
}

export async function probeComponent(text: string, layer: string, componentIdx: number): Promise<ComponentProbeResult> {
    return fetchJson<ComponentProbeResult>(`${API_URL}/api/activation_contexts/probe`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, layer, component_idx: componentIdx }),
    });
}
