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
    nComponentLayerKeys: number | null;
    vocabSize: number | null;
    ciThreshold: number | null;
};

export async function getDatasetAttributionMetadata(): Promise<DatasetAttributionMetadata> {
    const data = await fetchJson<{
        available: boolean;
        n_batches_processed: number | null;
        n_tokens_processed: number | null;
        n_component_layer_keys: number | null;
        vocab_size: number | null;
        ci_threshold: number | null;
    }>(`${API_URL}/api/dataset_attributions/metadata`);

    return {
        available: data.available,
        nBatchesProcessed: data.n_batches_processed,
        nTokensProcessed: data.n_tokens_processed,
        nComponentLayerKeys: data.n_component_layer_keys,
        vocabSize: data.vocab_size,
        ciThreshold: data.ci_threshold,
    };
}

export type ComponentAttributions = {
    positiveSources: DatasetAttributionEntry[];
    negativeSources: DatasetAttributionEntry[];
    positiveTargets: DatasetAttributionEntry[];
    negativeTargets: DatasetAttributionEntry[];
};

function mapEntries(
    entries: Array<{ component_key: string; layer: string; component_idx: number; value: number }>,
): DatasetAttributionEntry[] {
    return entries.map((e) => ({
        componentKey: e.component_key,
        layer: e.layer,
        componentIdx: e.component_idx,
        value: e.value,
    }));
}

export async function getComponentAttributions(
    layer: string,
    componentIdx: number,
    k: number = 10,
): Promise<ComponentAttributions> {
    const url = new URL(`${API_URL}/api/dataset_attributions/${encodeURIComponent(layer)}/${componentIdx}`);
    url.searchParams.set("k", String(k));

    const data = await fetchJson<{
        positive_sources: Array<{ component_key: string; layer: string; component_idx: number; value: number }>;
        negative_sources: Array<{ component_key: string; layer: string; component_idx: number; value: number }>;
        positive_targets: Array<{ component_key: string; layer: string; component_idx: number; value: number }>;
        negative_targets: Array<{ component_key: string; layer: string; component_idx: number; value: number }>;
    }>(url.toString());

    return {
        positiveSources: mapEntries(data.positive_sources),
        negativeSources: mapEntries(data.negative_sources),
        positiveTargets: mapEntries(data.positive_targets),
        negativeTargets: mapEntries(data.negative_targets),
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
    k: number = 10,
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
