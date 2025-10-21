export const API_URL = "http://localhost:8001";

type Layer = string;

type TokenComponentMaskIndices = number[];
export type ComponentMask = Record<Layer, TokenComponentMaskIndices[]>;

export type ComponentAblationRequest = {
    prompt_id: string;
    component_mask: Record<Layer, TokenComponentMaskIndices[]>;
};

const apiUrl: string = API_URL;

export type TrainRunDTO = {
    wandb_path: string;
    component_layers: string[];
    available_cluster_runs: string[];
    config: Record<string, any>;
};

export type ClusteringShape = {
    module_component_assignments: Record<string, number[]>;
    module_component_groups: Record<string, number[][]>;
};

export type ClusterRunDTO = {
    wandb_path: string;
    iteration: number;
    clustering_shape: ClusteringShape;
};

export type Status =
    | { train_run: null; cluster_run: null }
    | { train_run: TrainRunDTO; cluster_run: null }
    | { train_run: TrainRunDTO; cluster_run: ClusterRunDTO };

export async function getStatus(): Promise<Status> {
    const response = await fetch(`${apiUrl}/status`);
    const data = await response.json();
    return data;
}

export type AvailablePrompt = {
    index: number;
    full_text: string;
};

export async function getAvailablePrompts(): Promise<AvailablePrompt[]> {
    const response = await fetch(`${apiUrl}/available_prompts`, {
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
    component_agg_cis: number[];
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

export async function runPrompt(prompt: string): Promise<RunPromptResponse> {
    const response = await fetch(`${apiUrl}/run`, {
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

export async function runPromptByIndex(datasetIndex: number): Promise<RunPromptResponse> {
    const response = await fetch(`${apiUrl}/run_prompt/${datasetIndex}`, {
        method: "POST"
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to run prompt by index");
    }

    return response.json();
}

export type TokenAblationStats = {
    original_active_count: number;
    ablated_count: number;
    ablated_magnitude: number;
};

export type LayerAblationStats = {
    module: string;
    token_stats: TokenAblationStats[];
};

export type AblationStats = {
    layer_stats: LayerAblationStats[];
};

export type InterventionResponse = {
    token_logits: OutputTokenLogit[][];
    ablation_stats: AblationStats;
};

export async function applyMaskAsAblation(
    promptId: string,
    maskOverrideId: string
): Promise<InterventionResponse> {
    const response = await fetch(`${apiUrl}/apply_mask`, {
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

export type MaskOverrideDTO = {
    id: string;
    description: string | null;
    layer: string;
    combined_mask: SparseVector;
};

export async function getMaskOverrides(): Promise<MaskOverrideDTO[]> {
    const response = await fetch(`${apiUrl}/mask_overrides`, {
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

export async function ablateComponents(
    promptId: string,
    componentMask: ComponentMask
): Promise<InterventionResponse> {
    const req: ComponentAblationRequest = {
        prompt_id: promptId,
        component_mask: componentMask
    };

    console.log("ablateComponents", JSON.stringify(req));

    const response = await fetch(`${apiUrl}/ablate_components`, {
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

export type Run = {
    id: string;
    url: string;
};

export async function getRuns(): Promise<Run[]> {
    const response = await fetch(`${apiUrl}/runs`, {
        method: "GET",
        headers: {
            "Content-Type": "application/json"
        }
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to get runs");
    }
    return response.json();
}

export async function loadRun(wandbRunId: string): Promise<void> {
    const response = await fetch(`${apiUrl}/runs/load/${wandbRunId}`, {
        method: "POST"
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to load run");
    }
}

export type CosineSimilarityData = {
    input_singular_vectors: number[][]; // 2D array for pairwise cosine similarities
    output_singular_vectors: number[][]; // 2D array for pairwise cosine similarities
    component_indices: number[]; // indices corresponding to rows/cols
};

export async function getCosineSimilarities(
    layer: string,
    componentIdx: number
): Promise<CosineSimilarityData> {
    const url = `${apiUrl}/cosine_similarities/${layer}/${componentIdx}`;
    const response = await fetch(url, { method: "GET" });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to get cosine similarities");
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
    mask_override: MaskOverrideDTO;
};

export async function combineMasks(req: CombineMasksRequest): Promise<CombineMasksResponse> {
    const response = await fetch(`${apiUrl}/combine_masks`, {
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
    const response = await fetch(`${apiUrl}/simulate_merge`, {
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

export type ActivationContext = {
    raw_text: string;
    offset_mapping: [number, number][];
    token_ci_values: number[];
    active_position: number;
    ci_value: number;
};

export type SubcomponentActivationContexts = {
    subcomponent_idx: number;
    examples: ActivationContext[];
    token_densities: TokenDensity[];
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
    max_examples_per_subcomponent: number;
    n_batches: number;
    batch_size: number;
    n_tokens_either_side: number;
};

export async function getSubcomponentActivationContexts(
    config: ActivationContextsConfig
): Promise<ModelActivationContexts> {
    const url = new URL(`${apiUrl}/activation_contexts/subcomponents`);
    for (const [key, value] of Object.entries(config)) {
        url.searchParams.set(key, String(value));
    }
    const response = await fetch(url.toString(), { method: "GET" });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to get layer activation contexts");
    }
    return response.json();
}

// export async function getSubcomponentActivationContexts(
//     subcomponentIdx: number,
//     layer: string
// ): Promise<ActivationContext[]> {
//     const response = await fetch(
//         `${apiUrl}/activation_contexts/${layer}/subcomponents/${subcomponentIdx}`,
//         { method: "GET" }
//     );

//     if (!response.ok) {
//         const error = await response.json();
//         throw new Error(error.detail || "Failed to get activation contexts");
//     }

//     return response.json();
// }

export type ClusterComponentDTO = {
    module: string;
    index: number;
    label?: string;
};

export type ClusterStatsDTO = Record<string, any>;

export type ClusterDataDTO = {
    cluster_hash: string;
    components: ClusterComponentDTO[];
    stats: ClusterStatsDTO;
    criterion_samples: Record<string, string[]>;
};

export type TextSampleDTO = {
    text_hash: string;
    full_text: string;
    tokens: string[];
};

export type ActivationBatchDTO = {
    cluster_id: {
        clustering_run: string;
        iteration: number;
        cluster_label: number;
        hash: string;
    };
    text_hashes: string[];
    activations: number[][];
};

export type ClusterDashboardResponse = {
    clusters: ClusterDataDTO[];
    text_samples: TextSampleDTO[];
    activation_batch: ActivationBatchDTO;
    activations_map: Record<string, number>;
    model_info: Record<string, any>;
    iteration: number;
    run_path: string;
};

type DashboardQueryParams = {
    iteration: number;
    n_samples: number;
    n_batches: number;
    batch_size: number;
    context_length: number;
    signal?: AbortSignal;
};

export async function loadClusterRun(wandbRunPath: string, iteration: number): Promise<void> {
    const response = await fetch(`${apiUrl}/cluster-runs/load/${wandbRunPath}/${iteration}`, {
        method: "POST"
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to load cluster run");
    }
}

export async function getClusterDashboardData(
    params: DashboardQueryParams
): Promise<ClusterDashboardResponse> {
    const { signal, ...rest } = params;

    const url = new URL(`${apiUrl}/cluster-dashboard/data`);
    for (const [key, value] of Object.entries(rest)) {
        url.searchParams.set(key, String(value));
    }

    const response = await fetch(url.toString(), { method: "GET" });
    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || "Failed to load cluster dashboard data");
    }

    return response.json();
}
