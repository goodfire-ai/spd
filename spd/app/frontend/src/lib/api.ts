const API_URL = "http://localhost:8000";

export type StatusDTO = {
    loaded: boolean;
    run_id: string | null;
    prompt: string | null;
};

export type OutputTokenLogit = {
    token: string;
    logit: number;
    probability: number;
};

export type TokenCIs = {
    l0: number;
    component_cis: number[];
    indices: number[];
};

export type CosineSimilarityData = {
    input_singular_vectors: number[][]; // 2D array for pairwise cosine similarities
    output_singular_vectors: number[][]; // 2D array for pairwise cosine similarities
    component_indices: number[]; // indices corresponding to rows/cols
};

export type LayerCIs = {
    module: string;
    token_cis: TokenCIs[];
};

export type RunPromptResponse = {
    prompt_tokens: string[];
    layer_cis: LayerCIs[];
    full_run_token_logits: OutputTokenLogit[][];
    ci_masked_token_logits: OutputTokenLogit[][];
};

export type ComponentMask = Record<string, number[][]>;

export type ModifyComponentsResponse = {
    token_logits: OutputTokenLogit[][];
};

class ApiClient {
    constructor(private apiUrl: string = API_URL) {}

    async getStatus(): Promise<StatusDTO> {
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

    async runRandomPrompt(): Promise<RunPromptResponse> {
        const response = await fetch(`${this.apiUrl}/run_random`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            }
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Failed to run random prompt");
        }

        return response.json();
    }

    async ablateComponents(componentMask: ComponentMask): Promise<ModifyComponentsResponse> {
        console.log(
            "ablate",
            JSON.stringify({
                component_mask: componentMask
            })
        );
        const response = await fetch(`${this.apiUrl}/ablate`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
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
        const response = await fetch(`${this.apiUrl}/load`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ wandb_run_id: wandbRunId })
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Failed to load run");
        }
    }

    async getCosineSimilarities(
        layer: string,
        tokenIdx: number
    ): Promise<CosineSimilarityData> {
        const params = new URLSearchParams({
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
}

export const api = new ApiClient();
