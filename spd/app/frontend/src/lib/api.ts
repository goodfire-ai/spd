const API_URL = "http://localhost:8000";

export type SparseVector = {
    l0: number;
    indices: number[];
    values: number[];
};

export type Status = {
    loaded: boolean;
    run_id: string | null;
    prompt: string | null;
};

export type OutputTokenLogit = {
    token: string;
    logit: number;
    probability: number;
};

export type CosineSimilarityData = {
    input_singular_vectors: number[][]; // 2D array for pairwise cosine similarities
    output_singular_vectors: number[][]; // 2D array for pairwise cosine similarities
    component_indices: number[]; // indices corresponding to rows/cols
};

export type LayerCIs = {
    module: string;
    token_cis: SparseVector[];
};

export type RunPromptResponse = {
    prompt_id: string;
    prompt_tokens: string[];
    layer_cis: LayerCIs[];
    full_run_token_logits: OutputTokenLogit[][];
    ci_masked_token_logits: OutputTokenLogit[][];
};

export type ComponentMask = Record<string, number[][]>;

export type ModifyComponentsResponse = {
    token_logits: OutputTokenLogit[][];
};

export type MaskOverrideDTO = {
    id: string;
    description: string | null;
    layer: string;
    combined_mask: SparseVector;
};

export type CombineMasksRequest = {
    prompt_id: string;
    layer: string;
    token_indices: number[]; // List of token positions to combine
    description?: string;
};

export type CombineMasksResponse = {
    mask_id: string;
    mask_override: MaskOverrideDTO;
};

export type SimulateMergeRequest = {
    prompt_id: string;
    layer: string;
    token_indices: number[];
};

export type SimulateMergeResponse = {
    l0: number;
    jacc: number;
};

export type ActivationContext = {
    raw_text: string;
    offset_mapping: [number, number][];
    token_ci_values: number[];
    active_position: number;
    ci_value: number;
};

export type ComponentActivationContextsResponse = {
    component_idx: number;
    layer: string;
    examples: ActivationContext[];
};

class ApiClient {
    constructor(private apiUrl: string = API_URL) {}

    async getStatus(): Promise<Status> {
        const response = await fetch(`${this.apiUrl}/status`);
        return response.json();
    }

    async runPrompt(prompt: string): Promise<RunPromptResponse> {
        const response = await fetch(`${this.apiUrl}/run`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ prompt })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Failed to run prompt");
        }

        const data = await response.json();
        data.layer_cis.reverse();
        return data;
    }

    async getAvailablePrompts(): Promise<{ index: number; text: string; full_text: string }[]> {
        const response = await fetch(`${this.apiUrl}/available_prompts`, {
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

    async runPromptByIndex(datasetIndex: number): Promise<RunPromptResponse> {
        const response = await fetch(`${this.apiUrl}/run_prompt/${datasetIndex}`, {
            method: "POST"
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Failed to run prompt by index");
        }

        return response.json();
    }

    async applyMaskAsAblation(
        promptId: string,
        maskOverrideId: string
    ): Promise<ModifyComponentsResponse> {
        const response = await fetch(`${this.apiUrl}/apply_mask`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                prompt_id: promptId,
                mask_override_id: maskOverrideId
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Failed to apply mask as ablation");
        }

        return response.json();
    }

    async getMaskOverrides(): Promise<MaskOverrideDTO[]> {
        const response = await fetch(`${this.apiUrl}/mask_overrides`, {
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

    async ablateComponents(
        promptId: string,
        componentMask: ComponentMask
    ): Promise<ModifyComponentsResponse> {
        console.log(
            "ablate",
            JSON.stringify({
                prompt_id: promptId,
                component_mask: componentMask
            })
        );
        const response = await fetch(`${this.apiUrl}/ablate`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                prompt_id: promptId,
                component_mask: componentMask
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Failed to modify components");
        }

        return response.json();
    }

    async loadRun(wandbRunId: string): Promise<void> {
        const response = await fetch(`${this.apiUrl}/load/${wandbRunId}`, {
            method: "POST"
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Failed to load run");
        }
    }

    async getCosineSimilarities(
        promptId: string,
        layer: string,
        tokenIdx: number
    ): Promise<CosineSimilarityData> {
        const params = new URLSearchParams({
            prompt_id: promptId,
            layer,
            token_idx: tokenIdx.toString()
        });
        const response = await fetch(`${this.apiUrl}/cosine_similarities?${params}`, {
            method: "GET",
            headers: {
                "Content-Type": "application/json"
            }
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Failed to get cosine similarities");
        }

        return response.json();
    }

    async combineMasks(
        promptId: string,
        layer: string,
        tokenIndices: number[],
        description?: string
    ): Promise<CombineMasksResponse> {
        const response = await fetch(`${this.apiUrl}/combine_masks`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                prompt_id: promptId,
                layer: layer,
                token_indices: tokenIndices,
                description: description
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Failed to combine masks");
        }

        return response.json();
    }

    async simulateMerge(
        promptId: string,
        layer: string,
        tokenIndices: number[]
    ): Promise<SimulateMergeResponse> {
        const response = await fetch(`${this.apiUrl}/simulate_merge`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                prompt_id: promptId,
                layer: layer,
                token_indices: tokenIndices
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Failed to simulate merge");
        }

        return response.json();
    }

    async getComponentActivationContexts(
        componentId: number,
        layer: string
    ): Promise<ComponentActivationContextsResponse> {
        const response = await fetch(
            `${this.apiUrl}/component_activation_contexts/${layer}/${componentId}`,
            {
                method: "GET",
                headers: {
                    "Content-Type": "application/json"
                }
            }
        );

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Failed to get activation contexts");
        }

        return response.json();
    }
}

export const api = new ApiClient();
