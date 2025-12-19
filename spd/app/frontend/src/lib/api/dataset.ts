/**
 * API client for /api/dataset endpoints.
 */

import { API_URL } from "./index";

export type DatasetSearchResult = {
    story: string;
    occurrence_count: number;
    topic: string | null;
    theme: string | null;
};

export type DatasetSearchMetadata = {
    query: string;
    split: string;
    total_results: number;
    search_time_seconds: number;
};

export type DatasetSearchPage = {
    results: DatasetSearchResult[];
    page: number;
    page_size: number;
    total_results: number;
    total_pages: number;
};

export async function searchDataset(query: string, split: string): Promise<DatasetSearchMetadata> {
    const url = new URL(`${API_URL}/api/dataset/search`);
    url.searchParams.set("query", query);
    url.searchParams.set("split", split);

    const response = await fetch(url.toString(), { method: "POST" });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to search dataset");
    }

    return (await response.json()) as DatasetSearchMetadata;
}

export async function getDatasetSearchPage(page: number, pageSize: number): Promise<DatasetSearchPage> {
    const url = new URL(`${API_URL}/api/dataset/results`);
    url.searchParams.set("page", String(page));
    url.searchParams.set("page_size", String(pageSize));

    const response = await fetch(url.toString(), { method: "GET" });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to get search results");
    }

    return (await response.json()) as DatasetSearchPage;
}
