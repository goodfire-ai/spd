export const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export type TrainRun = {
    wandb_path: string;
    config: Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
};

export type Status = { train_run: TrainRun | null };

export async function getStatus(): Promise<Status> {
    const response = await fetch(`${API_URL}/status`);
    const data = await response.json();
    return data;
}

export async function loadRun(wandbRunPath: string): Promise<void> {
    const url = new URL(`${API_URL}/runs/load`);
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
    token_densities: TokenDensity[];
    mean_ci: number;
};

export type ModelActivationContexts = {
    layers: Record<string, SubcomponentActivationContexts[]>;
};

export type TokenDensity = {
    token: string;
    density: number;
};

export type ActivationContextsConfig = {
    importance_threshold: number;
    topk_examples: number;
    n_batches: number;
    batch_size: number;
    n_tokens_either_side: number;
};

export async function getSubcomponentActivationContexts(
    config: ActivationContextsConfig
): Promise<ModelActivationContexts> {
    const url = new URL(`${API_URL}/activation_contexts/subcomponents`);
    for (const [key, value] of Object.entries(config)) {
        url.searchParams.set(key, String(value));
    }
    const response = await fetch(url.toString(), { method: "GET" });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to get layer activation contexts");
    }
    const payload = (await response.json()) as unknown as ModelActivationContexts;
    for (const layer of Object.keys(payload.layers)) {
        for (const subcomponent of payload.layers[layer]) {
            for (const example of subcomponent.examples) {
                example.__id = crypto.randomUUID();
            }
        }
    }
    return payload;
}
