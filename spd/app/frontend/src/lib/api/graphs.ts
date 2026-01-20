/**
 * API client for /api/graphs endpoints.
 */

import type { GraphData, TokenizeResult, TokenInfo } from "../promptAttributionsTypes";
import { buildEdgeIndexes } from "../promptAttributionsTypes";
import { API_URL, ApiError, fetchJson } from "./index";

export type NormalizeType = "none" | "target" | "layer";

export type GraphProgress = {
    current: number;
    total: number;
    stage: string;
};

export type ComputeGraphParams = {
    promptId: number;
    normalize: NormalizeType;
    ciThreshold: number;
    /** If provided, only include these nodes in the graph (creates manual graph) */
    includedNodes?: string[];
};

/**
 * Parse SSE stream and return GraphData result.
 * Handles progress updates, errors, and completion messages.
 */
async function parseGraphSSEStream(
    response: Response,
    onProgress?: (progress: GraphProgress) => void,
): Promise<GraphData> {
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
                throw new ApiError(data.error, 500);
            } else if (data.type === "complete") {
                const { edgesBySource, edgesByTarget } = buildEdgeIndexes(data.data.edges);
                result = { ...data.data, edgesBySource, edgesByTarget };
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

export async function computeGraphStreaming(
    params: ComputeGraphParams,
    onProgress?: (progress: GraphProgress) => void,
): Promise<GraphData> {
    const url = new URL(`${API_URL}/api/graphs`);
    url.searchParams.set("prompt_id", String(params.promptId));
    url.searchParams.set("normalize", String(params.normalize));
    url.searchParams.set("ci_threshold", String(params.ciThreshold));
    if (params.includedNodes !== undefined) {
        url.searchParams.set("included_nodes", JSON.stringify(params.includedNodes));
    }

    const response = await fetch(url.toString(), { method: "POST" });
    if (!response.ok) {
        const error = await response.json();
        throw new ApiError(error.detail || `HTTP ${response.status}`, response.status);
    }

    return parseGraphSSEStream(response, onProgress);
}

export type MaskType = "stochastic" | "ci";

export type ComputeGraphOptimizedParams = {
    promptId: number;
    impMinCoeff: number;
    steps: number;
    pnorm: number;
    beta: number;
    normalize: NormalizeType;
    outputProbThreshold: number;
    ciThreshold: number;
    labelToken?: number;
    ceLossCoeff?: number;
    klLossCoeff?: number;
    maskType: MaskType;
};

export async function computeGraphOptimizedStreaming(
    params: ComputeGraphOptimizedParams,
    onProgress?: (progress: GraphProgress) => void,
): Promise<GraphData> {
    const url = new URL(`${API_URL}/api/graphs/optimized/stream`);
    url.searchParams.set("prompt_id", String(params.promptId));
    url.searchParams.set("imp_min_coeff", String(params.impMinCoeff));
    url.searchParams.set("steps", String(params.steps));
    url.searchParams.set("pnorm", String(params.pnorm));
    url.searchParams.set("beta", String(params.beta));
    url.searchParams.set("normalize", String(params.normalize));
    url.searchParams.set("output_prob_threshold", String(params.outputProbThreshold));
    url.searchParams.set("ci_threshold", String(params.ciThreshold));

    if (params.labelToken !== undefined) {
        url.searchParams.set("label_token", String(params.labelToken));
    }
    if (params.ceLossCoeff !== undefined) {
        url.searchParams.set("ce_loss_coeff", String(params.ceLossCoeff));
    }
    if (params.klLossCoeff !== undefined) {
        url.searchParams.set("kl_loss_coeff", String(params.klLossCoeff));
    }
    url.searchParams.set("mask_type", params.maskType);

    const response = await fetch(url.toString(), { method: "POST" });
    if (!response.ok) {
        const error = await response.json();
        throw new ApiError(error.detail || `HTTP ${response.status}`, response.status);
    }

    return parseGraphSSEStream(response, onProgress);
}

export async function getGraphs(promptId: number, normalize: NormalizeType, ciThreshold: number): Promise<GraphData[]> {
    const url = new URL(`${API_URL}/api/graphs/${promptId}`);
    url.searchParams.set("normalize", normalize);
    url.searchParams.set("ci_threshold", String(ciThreshold));
    const graphs = await fetchJson<Omit<GraphData, "edgesBySource" | "edgesByTarget">[]>(url.toString());
    return graphs.map((g) => {
        const { edgesBySource, edgesByTarget } = buildEdgeIndexes(g.edges);
        return { ...g, edgesBySource, edgesByTarget };
    });
}

export async function tokenizeText(text: string): Promise<TokenizeResult> {
    const url = new URL(`${API_URL}/api/graphs/tokenize`);
    url.searchParams.set("text", text);
    return fetchJson<TokenizeResult>(url.toString(), { method: "POST" });
}

export async function getAllTokens(): Promise<TokenInfo[]> {
    const response = await fetchJson<{ tokens: TokenInfo[] }>(`${API_URL}/api/graphs/tokens`);
    return response.tokens;
}
