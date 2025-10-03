export const API_URL = "http://localhost:8000";

type Layer = string;

type TokenComponentMaskIndices = number[];
export type ComponentMask = Record<Layer, TokenComponentMaskIndices[]>;

export type ComponentAblationRequest = {
    prompt_id: string;
    component_mask: Record<Layer, TokenComponentMaskIndices[]>;
};

const apiUrl: string = API_URL;

export type Status = {
    loaded: boolean;
    run_id: string | null;
    component_layers: string[]; // todo put me somewhere else
};
export async function getStatus(): Promise<Status> {
    const response = await fetch(`${apiUrl}/status`);
    return response.json();
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

export type Component = {
    index: number;
    subcomponent_indices: number[];
};

export type MatrixCausalImportances = {
    subcomponent_cis_sparse: SparseVector;
    subcomponent_cis: number[];
    component_agg_cis: number[];
    components: Component[];
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

export type InterventionResponse = {
    token_logits: OutputTokenLogit[][];
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
    console.log(
        "ablateComponents",
        JSON.stringify({
            prompt_id: promptId,
            component_mask: componentMask
        })
    );

    const response = await fetch(`${apiUrl}/ablate_components`, {
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

export type ClusterDashboardDataDirs = {
    dirs: string[];
    latest: string | null;
};

export async function getClusterDashboardDataDirs(
    runId?: string
): Promise<ClusterDashboardDataDirs> {
    const url = new URL(`${apiUrl}/dashboard/data-dirs`);
    if (runId) url.searchParams.set("run_id", runId);
    const response = await fetch(url.toString(), { method: "GET" });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to get cluster dashboard data dirs");
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

export type SimulateMergeResponse = {
    l0: number;
    jacc: number;
};

export async function simulateMerge(
    promptId: string,
    layer: string,
    tokenIndices: number[]
): Promise<SimulateMergeResponse> {
    const response = await fetch(`${apiUrl}/simulate_merge`, {
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
};

export async function getLayerActivationContexts(
    layer: string,
    signal?: AbortSignal
): Promise<SubcomponentActivationContexts[]> {
    const response = await fetch(`${apiUrl}/activation_contexts/${layer}/subcomponents`, {
        method: "GET",
        signal
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to get layer activation contexts");
    }
    return response.json();
}

export async function getSubcomponentActivationContexts(
    subcomponentIdx: number,
    layer: string
): Promise<ActivationContext[]> {
    const response = await fetch(
        `${apiUrl}/activation_contexts/${layer}/subcomponents/${subcomponentIdx}`,
        { method: "GET" }
    );

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to get activation contexts");
    }

    return response.json();
}

// ---- Demo data helpers (frontend-only scaffolding) ----

export type ClusterSummaryRow = {
    id: number;
    clusterHash: string;
    componentCount: number;
    modules: string[];
};

export type ClusterDetailData = {
    cluster_hash: string;
    components: { module: string; index: number; label?: string }[];
    stats?: Record<string, any>;
    criterion_samples?: Record<string, string[]>;
};

export async function fetchText(url: string): Promise<string> {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.text();
}

export function getDemoClusterRows(n: number = 20): ClusterSummaryRow[] {
    const modulePool = Array.from({ length: 4 }).flatMap((_, i) => [
        `model.layers.${i}.self_attn.q_proj`,
        `model.layers.${i}.self_attn.k_proj`,
        `model.layers.${i}.self_attn.v_proj`,
        `model.layers.${i}.self_attn.o_proj`,
        `model.layers.${i}.mlp.gate_proj`,
        `model.layers.${i}.mlp.up_proj`,
        `model.layers.${i}.mlp.down_proj`
    ]);
    const rand = (a: number, b: number) => Math.floor(Math.random() * (b - a + 1)) + a;
    const out: ClusterSummaryRow[] = [];
    for (let i = 0; i < n; i++) {
        const comps = rand(1, 12);
        const mods = new Set<string>();
        const picks = rand(1, 4);
        for (let k = 0; k < picks; k++) mods.add(modulePool[rand(0, modulePool.length - 1)]);
        out.push({
            id: i,
            clusterHash: `demo-0-${i}`,
            componentCount: comps,
            modules: Array.from(mods)
        });
    }
    return out;
}

export function getDemoClusterData(clusterHash: string): ClusterDetailData {
    const idNum = parseInt(clusterHash.split("-").pop() || "0");
    // seeded PRNG (mulberry32-like)
    let seed = (0x9e3779b9 ^ idNum) >>> 0;
    const rand = () => {
        seed |= 0;
        seed = (seed + 0x6d2b79f5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };

    // choose number of layers between 12 and 36
    const numLayers = 4;
    const sublayers = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj"
    ];

    const components: { module: string; index: number }[] = [];
    let nextIndex = 0;
    for (let layer = 0; layer < numLayers; layer++) {
        for (const sub of sublayers) {
            const p = 0.08 + 0.22 * rand();
            const n = rand() < p ? 1 + Math.floor(rand() * 3) : 0; // 0-3 comps
            for (let k = 0; k < n; k++) {
                components.push({ module: `model.layers.${layer}.${sub}`, index: nextIndex++ });
            }
        }
    }

    const mkBins = (len = 10, base = 5, skew = rand() * 2 - 1) => {
        const bins: number[] = [];
        for (let i = 0; i < len; i++) {
            const t = i / (len - 1);
            const shape = skew >= 0 ? Math.pow(1 - t, 1 + skew * 2) : Math.pow(t, 1 + -skew * 2);
            const noise = 0.3 + rand();
            bins.push(Math.max(0, Math.round(base * (0.6 + 1.8 * shape * noise))));
        }
        return bins;
    };

    const allBins = mkBins(10, 6);
    const posBins = mkBins(10, 5, rand() * 2 - 1);

    const tokensPool = [
        " the",
        " and",
        " to",
        " of",
        " in",
        " is",
        " it",
        " that",
        " as",
        " with",
        " you",
        " for"
    ];
    const topTokens = tokensPool
        .slice()
        .sort(() => rand() - 0.5)
        .slice(0, 5)
        .map((t, i) => ({ token: t, count: 15 + Math.floor(rand() * 60) - i * 5 }));
    const totalActivations = topTokens.reduce((a, b) => a + b.count, 0);

    const stats = {
        all_activations: {
            bin_edges: Array.from({ length: 11 }).map((_, i) => i / 10),
            bin_counts: allBins
        },
        "max_activation-max-16": {
            bin_edges: Array.from({ length: 11 }).map((_, i) => i / 10),
            bin_counts: mkBins(10, 4)
        },
        max_activation_position: {
            bin_edges: Array.from({ length: 11 }).map((_, i) => i / 10),
            bin_counts: posBins
        },
        token_activations: {
            top_tokens: topTokens,
            total_unique_tokens: topTokens.length,
            total_activations: totalActivations,
            entropy: 0.5 + rand() * 4,
            concentration_ratio: 0.3 + rand() * 0.5,
            activation_threshold: 0.5
        }
    } as Record<string, any>;

    return { cluster_hash: clusterHash, components, stats };
}

export type ClusterMap = Record<string, ClusterDetailData>;

export function getDemoClustersIndex(n: number): {
    rows: ClusterSummaryRow[];
    clusterMap: ClusterMap;
} {
    const rows = getDemoClusterRows(n);
    const clusterMap: ClusterMap = {};
    for (const r of rows) {
        clusterMap[r.clusterHash] = getDemoClusterData(r.clusterHash);
    }
    return { rows, clusterMap };
}

export async function getClustersIndex(
    absDir: string
): Promise<{ rows: ClusterSummaryRow[]; clusterMap: ClusterMap }> {
    const clustersUrl = `${absDir}/clusters.jsonl`;
    const text = await fetchText(clustersUrl);
    const clusterMap: ClusterMap = {};
    for (const line of text.trim().split("\n")) {
        if (!line) continue;
        const obj = JSON.parse(line) as ClusterDetailData;
        clusterMap[obj.cluster_hash] = obj;
    }
    const rows: ClusterSummaryRow[] = Object.entries(clusterMap).map(([clusterHash, cluster]) => {
        const modules = new Set<string>();
        cluster.components.forEach((c) => modules.add(c.module));
        const parts = clusterHash.split("-");
        const id = parseInt(parts[parts.length - 1]);
        return {
            id,
            clusterHash,
            componentCount: cluster.components.length,
            modules: Array.from(modules)
        } as ClusterSummaryRow;
    });
    rows.sort((a, b) => a.id - b.id);
    return { rows, clusterMap };
}
