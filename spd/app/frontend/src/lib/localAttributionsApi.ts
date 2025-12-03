/**
 * API client for local attributions (part of the unified SPD backend).
 */

import type {
    RunInfo,
    PromptPreview,
    PromptData,
    ActivationContextsSummary,
    ComponentDetail,
    SearchResult,
    ServerStatus,
    TokenizeResult,
} from "./localAttributionsTypes";
import { API_URL } from "./api";

export type { ServerStatus };

// Use the same API URL as the main app (unified backend on port 8000)
const LOCAL_ATTR_API_URL = API_URL;

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

// Run management

export async function listRuns(): Promise<RunInfo[]> {
    return fetchJson<RunInfo[]>(`${LOCAL_ATTR_API_URL}/api/runs`);
}

export async function loadRun(wandbPath: string): Promise<{ status: string; run_id?: number; wandb_path?: string }> {
    const url = new URL(`${LOCAL_ATTR_API_URL}/api/runs/load`);
    url.searchParams.set("wandb_path", wandbPath);
    return fetchJson(`${url}`, { method: "POST" });
}

export async function getStatus(): Promise<ServerStatus> {
    return fetchJson<ServerStatus>(`${LOCAL_ATTR_API_URL}/api/status`);
}

// Prompts

export async function listPrompts(): Promise<PromptPreview[]> {
    return fetchJson<PromptPreview[]>(`${LOCAL_ATTR_API_URL}/api/prompts`);
}

export type GetPromptParams = {
    maxMeanCI?: number;
    normalize?: boolean;
    ciThreshold?: number;
    outputProbThreshold?: number;
};

export async function getPrompt(promptId: number, params: GetPromptParams = {}): Promise<PromptData> {
    const url = new URL(`${LOCAL_ATTR_API_URL}/api/prompt/${promptId}`);

    if (params.maxMeanCI !== undefined) url.searchParams.set("max_mean_ci", String(params.maxMeanCI));
    if (params.normalize !== undefined) url.searchParams.set("normalize", String(params.normalize));
    if (params.ciThreshold !== undefined) url.searchParams.set("ci_threshold", String(params.ciThreshold));
    if (params.outputProbThreshold !== undefined)
        url.searchParams.set("output_prob_threshold", String(params.outputProbThreshold));

    return fetchJson<PromptData>(url.toString());
}

export type GetPromptOptimizedParams = GetPromptParams & {
    labelToken?: number;
    impMinCoeff?: number;
    ceLossCoeff?: number;
    steps?: number;
    lr?: number;
    pnorm?: number;
};

export async function getPromptOptimized(promptId: number, params: GetPromptOptimizedParams = {}): Promise<PromptData> {
    const url = new URL(`${LOCAL_ATTR_API_URL}/api/prompt/${promptId}/optimized`);

    if (params.labelToken !== undefined) url.searchParams.set("label_token", String(params.labelToken));
    if (params.impMinCoeff !== undefined) url.searchParams.set("imp_min_coeff", String(params.impMinCoeff));
    if (params.ceLossCoeff !== undefined) url.searchParams.set("ce_loss_coeff", String(params.ceLossCoeff));
    if (params.steps !== undefined) url.searchParams.set("steps", String(params.steps));
    if (params.lr !== undefined) url.searchParams.set("lr", String(params.lr));
    if (params.pnorm !== undefined) url.searchParams.set("pnorm", String(params.pnorm));
    if (params.maxMeanCI !== undefined) url.searchParams.set("max_mean_ci", String(params.maxMeanCI));
    if (params.normalize !== undefined) url.searchParams.set("normalize", String(params.normalize));
    if (params.ciThreshold !== undefined) url.searchParams.set("ci_threshold", String(params.ciThreshold));
    if (params.outputProbThreshold !== undefined)
        url.searchParams.set("output_prob_threshold", String(params.outputProbThreshold));

    return fetchJson<PromptData>(url.toString());
}

// Activation contexts

export async function getActivationContextsSummary(): Promise<ActivationContextsSummary> {
    return fetchJson<ActivationContextsSummary>(`${LOCAL_ATTR_API_URL}/api/activation_contexts/summary`);
}

export async function getComponentDetail(layer: string, componentIdx: number): Promise<ComponentDetail> {
    return fetchJson<ComponentDetail>(
        `${LOCAL_ATTR_API_URL}/api/activation_contexts/${encodeURIComponent(layer)}/${componentIdx}`,
    );
}

// Search

export async function searchPrompts(components: string[], mode: "all" | "any" = "all"): Promise<SearchResult> {
    const url = new URL(`${LOCAL_ATTR_API_URL}/api/search`);
    url.searchParams.set("components", components.join(","));
    url.searchParams.set("mode", mode);
    return fetchJson<SearchResult>(url.toString());
}

// Custom prompts

export async function tokenizeText(text: string): Promise<TokenizeResult> {
    const url = new URL(`${LOCAL_ATTR_API_URL}/api/tokenize`);
    url.searchParams.set("text", text);
    return fetchJson<TokenizeResult>(url.toString(), { method: "POST" });
}

export type ComputeCustomPromptParams = {
    tokenIds: number[];
    normalize?: boolean;
    ciThreshold?: number;
    outputProbThreshold?: number;
};

export async function computeCustomPrompt(params: ComputeCustomPromptParams): Promise<PromptData> {
    const url = new URL(`${LOCAL_ATTR_API_URL}/api/prompt/custom`);
    if (params.normalize !== undefined) url.searchParams.set("normalize", String(params.normalize));
    if (params.ciThreshold !== undefined) url.searchParams.set("ci_threshold", String(params.ciThreshold));
    if (params.outputProbThreshold !== undefined)
        url.searchParams.set("output_prob_threshold", String(params.outputProbThreshold));

    return fetchJson<PromptData>(url.toString(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token_ids: params.tokenIds }),
    });
}

// Prompt generation with CI harvesting

export type GeneratePromptsConfig = {
    nPrompts?: number;
    ciThreshold?: number;
    outputProbThreshold?: number;
};

export type GeneratePromptsResult = {
    prompts_added: number;
    total_prompts: number;
};

export async function generatePrompts(
    config: GeneratePromptsConfig,
    onProgress?: (progress: number, count: number) => void,
): Promise<GeneratePromptsResult> {
    const url = new URL(`${LOCAL_ATTR_API_URL}/api/prompts/generate`);
    if (config.nPrompts !== undefined) url.searchParams.set("n_prompts", String(config.nPrompts));
    if (config.ciThreshold !== undefined) url.searchParams.set("ci_threshold", String(config.ciThreshold));
    if (config.outputProbThreshold !== undefined)
        url.searchParams.set("output_prob_threshold", String(config.outputProbThreshold));

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
