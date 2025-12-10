// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const API_URL = (import.meta as any).env.VITE_API_URL || "http://localhost:8000";

export type LoadedRun = {
    id: number;
    wandb_path: string;
    config_yaml: string;
    has_activation_contexts: boolean;
    has_prompts: boolean;
    prompt_count: number;
    context_length: number;
    backend_user: string;
};

export async function getStatus(): Promise<LoadedRun | null> {
    const response = await fetch(`${API_URL}/api/status`);
    const data = await response.json();
    return data;
}

export async function getWhoami(): Promise<string> {
    const response = await fetch(`${API_URL}/api/whoami`);
    const data = await response.json();
    return data.user;
}

export async function loadRun(wandbRunPath: string, contextLength: number): Promise<void> {
    const url = new URL(`${API_URL}/api/runs/load`);
    // searchParams.set handles URL encoding automatically
    url.searchParams.set("wandb_path", wandbRunPath);
    url.searchParams.set("context_length", String(contextLength));
    const response = await fetch(url.toString(), { method: "POST" });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to load run");
    }
}

// Columnar data structure for efficiency
export type SubcomponentActivationContexts = {
    subcomponent_idx: number;
    mean_ci: number;
    // Examples - columnar arrays (n_examples, window_size)
    example_tokens: string[][]; // [n_examples][window_size]
    example_ci: number[][]; // [n_examples][window_size]
    example_active_pos: number[]; // [n_examples]
    example_active_ci: number[]; // [n_examples]
    // Token precision/recall - columnar arrays sorted by recall descending
    pr_tokens: string[]; // [n_unique_tokens]
    pr_recalls: number[]; // [n_unique_tokens]
    pr_precisions: number[]; // [n_unique_tokens]
};

export type ModelActivationContexts = {
    layers: Record<string, SubcomponentActivationContexts[]>;
};

export type ActivationContextsConfig = {
    importance_threshold: number;
    topk_examples: number;
    n_batches: number;
    batch_size: number;
    n_tokens_either_side: number;
    separation_tokens: number;
};

export type ProgressUpdate = {
    /** Progress as a float from 0.0 to 1.0 */
    progress: number;
};

// New types for lazy-loading
export type SubcomponentMetadata = {
    subcomponent_idx: number;
    mean_ci: number;
};

export type HarvestMetadata = {
    layers: Record<string, SubcomponentMetadata[]>;
};

// Full component detail (matches SubcomponentActivationContexts on backend)
export type ComponentDetail = SubcomponentActivationContexts;

// Streaming version with lazy-loading support
export async function getSubcomponentActivationContexts(
    config: ActivationContextsConfig,
    onProgress?: (progress: ProgressUpdate) => void,
): Promise<HarvestMetadata> {
    const url = new URL(`${API_URL}/api/activation_contexts/subcomponents`);
    for (const [key, value] of Object.entries(config)) {
        url.searchParams.set(key, String(value));
    }
    const response = await fetch(url.toString(), { method: "GET" });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to get layer activation contexts");
    }

    const reader = response.body?.getReader();
    if (!reader) {
        throw new Error("Response body is not readable");
    }

    const decoder = new TextDecoder();
    let buffer = "";
    let result: HarvestMetadata | null = null;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE messages (ending with \n\n)
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || ""; // Keep incomplete message in buffer

        for (const line of lines) {
            if (!line.trim() || !line.startsWith("data: ")) continue;

            const data = JSON.parse(line.substring(6)); // Remove "data: " prefix

            if (data.type === "progress" && onProgress) {
                onProgress({ progress: data.progress });
            } else if (data.type === "complete") {
                result = data.result as HarvestMetadata;
                // Close the reader early - we got what we need
                await reader.cancel();
                break;
            }
        }

        // Break out of outer loop if we got the result
        if (result) break;
    }

    if (!result) {
        throw new Error("No result received from stream");
    }

    return result;
}

// Lazy-load individual component data
export async function getComponentDetail(layer: string, componentIdx: number): Promise<ComponentDetail> {
    const url = `${API_URL}/api/activation_contexts/${encodeURIComponent(layer)}/${componentIdx}`;
    const response = await fetch(url, { method: "GET" });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || `Failed to get component ${componentIdx} for layer ${layer}`);
    }
    return (await response.json()) as ComponentDetail;
}

// Intervention types
import type {
    InterventionNode,
    InterventionResponse,
    InterventionRunSummary,
    RunInterventionRequest,
} from "./interventionTypes";

export async function runIntervention(
    text: string,
    nodes: InterventionNode[],
    topK: number = 10,
): Promise<InterventionResponse> {
    const response = await fetch(`${API_URL}/api/intervention`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, nodes, top_k: topK }),
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to run intervention");
    }
    return (await response.json()) as InterventionResponse;
}

/** Run an intervention and save the result */
export async function runAndSaveIntervention(request: RunInterventionRequest): Promise<InterventionRunSummary> {
    const response = await fetch(`${API_URL}/api/intervention/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to run intervention");
    }
    return (await response.json()) as InterventionRunSummary;
}

/** Get all intervention runs for a graph */
export async function getInterventionRuns(graphId: number): Promise<InterventionRunSummary[]> {
    const response = await fetch(`${API_URL}/api/intervention/runs/${graphId}`);
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to get intervention runs");
    }
    return (await response.json()) as InterventionRunSummary[];
}

/** Delete an intervention run */
export async function deleteInterventionRun(runId: number): Promise<void> {
    const response = await fetch(`${API_URL}/api/intervention/runs/${runId}`, {
        method: "DELETE",
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to delete intervention run");
    }
}

// =============================================================================
// Dataset Search
// =============================================================================

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
