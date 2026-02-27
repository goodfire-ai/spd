/**
 * API client for /api/dataset_attributions endpoints.
 */

import { apiUrl, fetchJson } from "./index";

export type DatasetAttributionEntry = {
    component_key: string;
    layer: string;
    component_idx: number;
    value: number;
};

export type ComponentAttributions = {
    positive_sources: DatasetAttributionEntry[];
    negative_sources: DatasetAttributionEntry[];
    positive_targets: DatasetAttributionEntry[];
    negative_targets: DatasetAttributionEntry[];
};

export type DatasetAttributionsMetadata = {
    available: boolean;
};

export async function getDatasetAttributionsMetadata(): Promise<DatasetAttributionsMetadata> {
    return fetchJson<DatasetAttributionsMetadata>(apiUrl("/api/dataset_attributions/metadata").toString());
}

export async function getComponentAttributions(
    layer: string,
    componentIdx: number,
    k: number = 10,
): Promise<ComponentAttributions> {
    const url = apiUrl(`/api/dataset_attributions/${encodeURIComponent(layer)}/${componentIdx}`);
    url.searchParams.set("k", String(k));
    return fetchJson<ComponentAttributions>(url.toString());
}
