/**
 * API client for local attributions (part of the unified SPD backend).
 */

import type {
    PromptPreview,
    GraphData,
    ActivationContextsSummary,
    ComponentDetail,
    ComponentCorrelations,
    TokenStats,
    SearchResult,
    TokenizeResult,
    ComponentProbeResult,
    TokenInfo,
} from "./localAttributionsTypes";
import { API_URL } from "./api";

class LocalAttributionsApiError extends Error {
    constructor(
        message: string,
        public status: number,
    ) {
        super(message);
        this.name = "LocalAttributionsApiError";
    }
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
    const response = await fetch(url, options);
    const data = await response.json();

    if (!response.ok) {
        throw new LocalAttributionsApiError(data.error || `HTTP ${response.status}`, response.status);
    }

    return data as T;
}

export async function listPrompts(): Promise<PromptPreview[]> {
    return fetchJson<PromptPreview[]>(`${API_URL}/api/prompts`);
}

export type GraphProgress = {
    current: number;
    total: number;
    stage: string;
};

// Activation contexts

export async function getActivationContextsSummary(): Promise<ActivationContextsSummary> {
    return fetchJson<ActivationContextsSummary>(`${API_URL}/api/activation_contexts/summary`);
}

export type ActivationContextsConfig = {
    importance_threshold: number;
    n_batches: number;
    batch_size: number;
    n_tokens_either_side: number;
    topk_examples: number;
    separation_tokens: number;
};

export async function getActivationContextsConfig(): Promise<ActivationContextsConfig | null> {
    return fetchJson<ActivationContextsConfig | null>(`${API_URL}/api/activation_contexts/config`);
}

export async function getComponentDetail(layer: string, componentIdx: number): Promise<ComponentDetail> {
    return fetchJson<ComponentDetail>(
        `${API_URL}/api/activation_contexts/${encodeURIComponent(layer)}/${componentIdx}`,
    );
}

// Search

export async function searchPrompts(components: string[], mode: "all" | "any" = "all"): Promise<SearchResult> {
    const url = new URL(`${API_URL}/api/prompts/search`);
    url.searchParams.set("components", components.join(","));
    url.searchParams.set("mode", mode);
    return fetchJson<SearchResult>(url.toString());
}

// Custom prompts

export async function createCustomPrompt(text: string): Promise<PromptPreview> {
    const url = new URL(`${API_URL}/api/prompts/custom`);
    url.searchParams.set("text", text);
    return fetchJson<PromptPreview>(url.toString(), { method: "POST" });
}

// Tokenize (preview only, doesn't persist)

export async function tokenizeText(text: string): Promise<TokenizeResult> {
    const url = new URL(`${API_URL}/api/graphs/tokenize`);
    url.searchParams.set("text", text);
    return fetchJson<TokenizeResult>(url.toString(), { method: "POST" });
}

export async function getAllTokens(): Promise<TokenInfo[]> {
    const response = await fetchJson<{ tokens: TokenInfo[] }>(`${API_URL}/api/graphs/tokens`);
    return response.tokens;
}

export type NormalizeType = "none" | "target" | "layer";

export type ComputeGraphParams = {
    promptId: number;
    normalize: NormalizeType;
    ciThreshold: number;
};

export async function computeGraphStreaming(
    params: ComputeGraphParams,
    onProgress?: (progress: GraphProgress) => void,
): Promise<GraphData> {
    const url = new URL(`${API_URL}/api/graphs`);
    url.searchParams.set("prompt_id", String(params.promptId));
    url.searchParams.set("normalize", String(params.normalize));
    url.searchParams.set("ci_threshold", String(params.ciThreshold));

    const response = await fetch(url.toString(), {
        method: "POST",
    });
    if (!response.ok) {
        const error = await response.json();
        throw new LocalAttributionsApiError(error.error || `HTTP ${response.status}`, response.status);
    }

    const reader = response.body?.getReader();
    if (!reader) {
        throw new Error("Response body is not readable");
    }

    const decoder = new TextDecoder();
    let buffer = "";
    let result: GraphData | null = null;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
            if (!line.trim() || !line.startsWith("data: ")) continue;

            const data = JSON.parse(line.substring(6));

            if (data.type === "progress" && onProgress) {
                onProgress({ current: data.current, total: data.total, stage: data.stage });
            } else if (data.type === "error") {
                throw new LocalAttributionsApiError(data.error, 500);
            } else if (data.type === "complete") {
                result = data.data;
                await reader.cancel();
                break;
            }
        }

        if (result) break;
    }

    if (!result) {
        throw new Error("No result received from stream");
    }

    return result;
}

export type ComputeGraphOptimizedParams = {
    promptId: number;
    labelToken: number;
    impMinCoeff: number;
    ceLossCoeff: number;
    steps: number;
    pnorm: number;
    normalize: NormalizeType;
    outputProbThreshold: number;
    ciThreshold: number;
};

export async function computeGraphOptimizedStreaming(
    params: ComputeGraphOptimizedParams,
    onProgress?: (progress: GraphProgress) => void,
): Promise<GraphData> {
    const url = new URL(`${API_URL}/api/graphs/optimized/stream`);
    url.searchParams.set("prompt_id", String(params.promptId));
    url.searchParams.set("label_token", String(params.labelToken));
    url.searchParams.set("imp_min_coeff", String(params.impMinCoeff));
    url.searchParams.set("ce_loss_coeff", String(params.ceLossCoeff));
    url.searchParams.set("steps", String(params.steps));
    url.searchParams.set("pnorm", String(params.pnorm));
    url.searchParams.set("normalize", String(params.normalize));
    url.searchParams.set("output_prob_threshold", String(params.outputProbThreshold));
    url.searchParams.set("ci_threshold", String(params.ciThreshold));

    const response = await fetch(url.toString(), {
        method: "POST",
    });
    if (!response.ok) {
        const error = await response.json();
        throw new LocalAttributionsApiError(error.error || `HTTP ${response.status}`, response.status);
    }

    const reader = response.body?.getReader();
    if (!reader) {
        throw new Error("Response body is not readable");
    }

    const decoder = new TextDecoder();
    let buffer = "";
    let result: GraphData | null = null;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
            if (!line.trim() || !line.startsWith("data: ")) continue;

            const data = JSON.parse(line.substring(6));

            if (data.type === "progress" && onProgress) {
                onProgress({ current: data.current, total: data.total, stage: data.stage });
            } else if (data.type === "error") {
                throw new LocalAttributionsApiError(data.error, 500);
            } else if (data.type === "complete") {
                result = data.data;
                await reader.cancel();
                break;
            }
        }

        if (result) break;
    }

    if (!result) {
        throw new Error("No result received from stream");
    }

    return result;
}

// Prompt generation with CI harvesting

export type GeneratePromptsConfig = {
    nPrompts: number;
};

export type GeneratePromptsResult = {
    prompts_added: number;
    total_prompts: number;
};

export async function generatePrompts(
    config: GeneratePromptsConfig,
    onProgress?: (progress: number, count: number) => void,
): Promise<GeneratePromptsResult> {
    const url = new URL(`${API_URL}/api/prompts/generate`);
    url.searchParams.set("n_prompts", String(config.nPrompts));

    const response = await fetch(url.toString(), { method: "POST" });
    if (!response.ok) {
        const error = await response.json();
        throw new LocalAttributionsApiError(error.detail || `HTTP ${response.status}`, response.status);
    }

    const reader = response.body?.getReader();
    if (!reader) {
        throw new Error("Response body is not readable");
    }

    const decoder = new TextDecoder();
    let buffer = "";
    let result: GeneratePromptsResult | null = null;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE messages
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
            if (!line.trim() || !line.startsWith("data: ")) continue;

            const data = JSON.parse(line.substring(6));

            if (data.type === "progress" && onProgress) {
                onProgress(data.progress, data.count);
            } else if (data.type === "complete") {
                result = { prompts_added: data.prompts_added, total_prompts: data.total_prompts };
                await reader.cancel();
                break;
            }
        }

        if (result) break;
    }

    if (!result) {
        throw new Error("No result received from stream");
    }

    return result;
}

// Fetch stored graphs for a prompt

export async function getGraphs(promptId: number, normalize: NormalizeType, ciThreshold: number): Promise<GraphData[]> {
    const url = new URL(`${API_URL}/api/graphs/${promptId}`);
    url.searchParams.set("normalize", normalize);
    url.searchParams.set("ci_threshold", String(ciThreshold));
    return fetchJson<GraphData[]>(url.toString());
}

// Probe component CI on custom text

export async function probeComponent(text: string, layer: string, componentIdx: number): Promise<ComponentProbeResult> {
    return fetchJson<ComponentProbeResult>(`${API_URL}/api/activation_contexts/probe`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, layer, component_idx: componentIdx }),
    });
}

// Component correlations

export async function getComponentCorrelations(
    layer: string,
    componentIdx: number,
    topK: number = 10,
): Promise<ComponentCorrelations | null> {
    const url = new URL(
        `${API_URL}/api/activation_contexts/correlations/${encodeURIComponent(layer)}/${componentIdx}`,
    );
    url.searchParams.set("top_k", String(topK));
    return fetchJson<ComponentCorrelations | null>(url.toString());
}

// Token stats from batch job (P/R/lift)
export async function getComponentTokenStats(
    layer: string,
    componentIdx: number,
    topK: number = 10,
): Promise<TokenStats | null> {
    const url = new URL(
        `${API_URL}/api/activation_contexts/token_stats/${encodeURIComponent(layer)}/${componentIdx}`,
    );
    url.searchParams.set("top_k", String(topK));
    return fetchJson<TokenStats | null>(url.toString());
}
