/**
 * API client for /api/dataset_attributions endpoints.
 */

import { API_URL, fetchJson } from "./index";

export type DatasetAttributionEntry = {
    componentKey: string;
    layer: string;
    componentIdx: number;
    value: number;
};

export type DatasetAttributionMetadata = {
    available: boolean;
    nBatchesProcessed: number | null;
    nTokensProcessed: number | null;
    nComponents: number | null;
    ciThreshold: number | null;
};

export async function getDatasetAttributionMetadata(): Promise<DatasetAttributionMetadata> {
    const data = await fetchJson<{
        available: boolean;
        n_batches_processed: number | null;
        n_tokens_processed: number | null;
        n_components: number | null;
        ci_threshold: number | null;
    }>(`${API_URL}/api/dataset_attributions/metadata`);

    return {
        available: data.available,
        nBatchesProcessed: data.n_batches_processed,
        nTokensProcessed: data.n_tokens_processed,
        nComponents: data.n_components,
        ciThreshold: data.ci_threshold,
    };
}

export async function getAttributionSources(
    layer: string,
    componentIdx: number,
    k: number = 10,
    sign: "positive" | "negative" = "positive",
): Promise<DatasetAttributionEntry[]> {
    const url = new URL(`${API_URL}/api/dataset_attributions/${encodeURIComponent(layer)}/${componentIdx}/sources`);
    url.searchParams.set("k", String(k));
    url.searchParams.set("sign", sign);

    const data = await fetchJson<
        Array<{
            component_key: string;
            layer: string;
            component_idx: number;
            value: number;
        }>
    >(url.toString());

    return data.map((entry) => ({
        componentKey: entry.component_key,
        layer: entry.layer,
        componentIdx: entry.component_idx,
        value: entry.value,
    }));
}

export async function getAttributionTargets(
    layer: string,
    componentIdx: number,
    k: number = 20,
    sign: "positive" | "negative" = "positive",
): Promise<DatasetAttributionEntry[]> {
    const url = new URL(`${API_URL}/api/dataset_attributions/${encodeURIComponent(layer)}/${componentIdx}/targets`);
    url.searchParams.set("k", String(k));
    url.searchParams.set("sign", sign);

    const data = await fetchJson<
        Array<{
            component_key: string;
            layer: string;
            component_idx: number;
            value: number;
        }>
    >(url.toString());

    return data.map((entry) => ({
        componentKey: entry.component_key,
        layer: entry.layer,
        componentIdx: entry.component_idx,
        value: entry.value,
    }));
}

export async function getAttributionBetween(
    sourceLayer: string,
    sourceIdx: number,
    targetLayer: string,
    targetIdx: number,
): Promise<number> {
    const url = `${API_URL}/api/dataset_attributions/between/${encodeURIComponent(sourceLayer)}/${sourceIdx}/${encodeURIComponent(targetLayer)}/${targetIdx}`;
    return fetchJson<number>(url);
}
