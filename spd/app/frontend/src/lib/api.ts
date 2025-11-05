// export const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
export const API_URL = "http://localhost:8000";

export type TrainRun = {
    wandb_path: string;
    config_yaml: string;
};

export type Status = { train_run: TrainRun | null };

export async function getStatus(): Promise<Status> {
    const response = await fetch(`${API_URL}/status`);
    const data = await response.json();
    return data;
}

export async function loadRun(wandbRunPath: string): Promise<void> {
    const url = new URL(`${API_URL}/runs/load`);
    // url-encode the wandb run path because it contains slashes
    const encodedWandbRunPath = encodeURIComponent(wandbRunPath);
    url.searchParams.set("wandb_run_path", encodedWandbRunPath);
    const response = await fetch(url.toString(), { method: "POST" });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to load run");
    }
}

export type ActivationContext = {
    token_strings: string[];
    token_ci_values: number[];
    active_position: number;
    ci_value: number;
    __id: string;
};

export type SubcomponentActivationContexts = {
    subcomponent_idx: number;
    examples: ActivationContext[];
    token_prs: TokenPR[];
    mean_ci: number;
};

export type ModelActivationContexts = {
    layers: Record<string, SubcomponentActivationContexts[]>;
};

export type TokenPR = {
    token: string;
    recall: number;
    precision: number;
};

export type ActivationContextsConfig = {
    importance_threshold: number;
    topk_examples: number;
    n_batches: number;
    batch_size: number;
    n_tokens_either_side: number;
};

export type ProgressUpdate = {
    current: number;
    total: number;
};

// New types for lazy-loading
export type SubcomponentMetadata = {
    subcomponent_idx: number;
    mean_ci: number;
};

export type HarvestMetadata = {
    harvest_id: string;
    layers: Record<string, SubcomponentMetadata[]>;
};

export type ComponentDetail = {
    examples: ActivationContext[];
    token_prs: TokenPR[];
    mean_ci: number;
};

// Streaming version with lazy-loading support
export async function getSubcomponentActivationContexts(
    config: ActivationContextsConfig,
    onProgress?: (progress: ProgressUpdate) => void
): Promise<HarvestMetadata> {
    const url = new URL(`${API_URL}/activation_contexts/subcomponents`);
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
                onProgress({ current: data.current, total: data.total });
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
export async function getComponentDetail(
    harvestId: string,
    layer: string,
    componentIdx: number
): Promise<ComponentDetail> {
    const url = `${API_URL}/activation_contexts/${encodeURIComponent(harvestId)}/${encodeURIComponent(layer)}/${componentIdx}`;
    const response = await fetch(url, { method: "GET" });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || `Failed to get component ${componentIdx} for layer ${layer}`);
    }

    const detail = (await response.json()) as ComponentDetail;

    // Add IDs to examples
    for (const example of detail.examples) {
        example.__id = crypto.randomUUID();
    }

    return detail;
}
