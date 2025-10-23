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

export async function loadRun(wandbRunId: string): Promise<void> {
    const response = await fetch(`${API_URL}/runs/load/${wandbRunId}`, {
        method: "POST"
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to load run");
    }
}

export type AvailablePrompt = {
    index: number;
    full_text: string;
};

export async function getAvailablePrompts(): Promise<AvailablePrompt[]> {
    const response = await fetch(`${API_URL}/available_prompts`, {
        method: "GET",
        headers: {
            "Content-Type": "application/json"
        }
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to get available prompts");
    }

    return response.json();
}

export type SparseVector = {
    l0: number;
    indices: number[];
    values: number[];
};

export type OutputTokenLogit = {
    token: string;
    logit: number;
    probability: number;
};

export type MatrixCausalImportances = {
    subcomponent_cis_sparse: SparseVector;
    subcomponent_cis: number[];
};

export type LayerCIs = {
    module: string;
    token_cis: MatrixCausalImportances[];
};

export type RunPromptResponse = {
    prompt_id: string;
    prompt_tokens: string[];
    layer_cis: LayerCIs[];
    full_run_token_logits: OutputTokenLogit[][];
    ci_masked_token_logits: OutputTokenLogit[][];
};

export async function runPromptByIndex(datasetIndex: number): Promise<RunPromptResponse> {
    const response = await fetch(`${API_URL}/run_prompt/${datasetIndex}`, {
        method: "POST"
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to run prompt by index");
    }

    return response.json();
}

export type SubcomponentMask = Record<string, number[][]>;

export type SubcomponentAblationRequest = {
    prompt_id: string;
    subcomponent_mask: SubcomponentMask;
};

export async function ablateSubcomponents(
    promptId: string,
    subcomponentMask: SubcomponentMask
): Promise<InterventionResponse> {
    const req: SubcomponentAblationRequest = {
        prompt_id: promptId,
        subcomponent_mask: subcomponentMask
    };

    const response = await fetch(`${API_URL}/ablate_subcomponents`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(req)
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to modify components");
    }

    return response.json();
}

export type SimulateMergeRequest = {
    prompt_id: string;
    layer: string;
    token_indices: number[];
};

export type SimulateMergeResponse = {
    l0: number;
    jacc: number;
};

export async function simulateMerge(req: SimulateMergeRequest): Promise<SimulateMergeResponse> {
    const response = await fetch(`${API_URL}/simulate_merge`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(req)
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to simulate merge");
    }

    return response.json();
}

export type CombineMasksRequest = {
    prompt_id: string;
    layer: string;
    token_indices: number[]; // List of token positions to combine
    description?: string;
};

export type CombineMasksResponse = {
    mask_id: string;
    mask: MaskOverride;
};

export async function combineMasks(req: CombineMasksRequest): Promise<CombineMasksResponse> {
    const response = await fetch(`${API_URL}/combine_masks`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(req)
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to combine masks");
    }

    return response.json();
}

export type MaskOverride = {
    id: string;
    description: string | null;
    layer: string;
    combined_mask: SparseVector;
};

export async function getMaskOverrides(): Promise<MaskOverride[]> {
    const response = await fetch(`${API_URL}/mask`, {
        method: "GET",
        headers: {
            "Content-Type": "application/json"
        }
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to get mask overrides");
    }

    return response.json();
}

export type TokenAblationEffect = {
    original_active_count: number;
    ablated_count: number;
    ablated_magnitude: number;
};

export type LayerAblationEffect = {
    module: string;
    token_abl_effect: TokenAblationEffect[];
};

export type AblationEffect = {
    layer_abl_effect: LayerAblationEffect[];
};

export type InterventionResponse = {
    token_logits: OutputTokenLogit[][];
    ablation_effect: AblationEffect;
};

export async function applyMaskAsAblation(
    promptId: string,
    maskOverrideId: string
): Promise<InterventionResponse> {
    const response = await fetch(`${API_URL}/apply_mask`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            prompt_id: promptId,
            mask_id: maskOverrideId
        })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to apply mask as ablation");
    }

    return response.json();
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
