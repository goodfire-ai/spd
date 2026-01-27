/**
 * API client for /api/activation_contexts endpoints.
 */

import type {
    ActivationContextsSummary,
    ComponentProbeResult,
    SubcomponentActivationContexts,
} from "../promptAttributionsTypes";
import { API_URL, fetchJson } from "./index";

export async function getActivationContextsSummary(): Promise<ActivationContextsSummary> {
    return fetchJson<ActivationContextsSummary>(`${API_URL}/api/activation_contexts/summary`);
}

/** Default limit for initial load - keeps payload small for fast initial render. */
const ACTIVATION_EXAMPLES_INITIAL_LIMIT = 30;

export async function getActivationContextDetail(
    layer: string,
    componentIdx: number,
    limit: number = ACTIVATION_EXAMPLES_INITIAL_LIMIT,
): Promise<SubcomponentActivationContexts> {
    return fetchJson<SubcomponentActivationContexts>(
        `${API_URL}/api/activation_contexts/${encodeURIComponent(layer)}/${componentIdx}?limit=${limit}`,
    );
}

export async function probeComponent(text: string, layer: string, componentIdx: number): Promise<ComponentProbeResult> {
    return fetchJson<ComponentProbeResult>(`${API_URL}/api/activation_contexts/probe`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, layer, component_idx: componentIdx }),
    });
}
