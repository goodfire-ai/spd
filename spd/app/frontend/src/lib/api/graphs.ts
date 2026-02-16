/**
 * API client for /api/graphs endpoints.
 */

import type { GraphData, TokenizeResponse, TokenInfo, GraphType, OptimizationResult } from "../promptAttributionsTypes";
import { buildEdgeIndexes } from "../promptAttributionsTypes";
import { setArchitecture } from "../layerAliasing";
import { apiUrl, ApiError, fetchJson } from "./index";

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
                // Extract all unique layer names from edges to detect architecture
                const layerNames = new Set<string>();
                for (const edge of data.data.edges) {
                    layerNames.add(edge.src.split(":")[0]);
                    layerNames.add(edge.tgt.split(":")[0]);
                }
                setArchitecture(Array.from(layerNames));

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

export async function computeGraphStream(
    params: ComputeGraphParams,
    onProgress?: (progress: GraphProgress) => void,
): Promise<GraphData> {
    const url = apiUrl("/api/graphs");
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
export type LossType = "ce" | "kl";

export type ComputeGraphOptimizedParams = {
    promptId: number;
    impMinCoeff: number;
    steps: number;
    pnorm: number;
    beta: number;
    normalize: NormalizeType;
    ciThreshold: number;
    maskType: MaskType;
    lossType: LossType;
    lossCoeff: number;
    lossPosition: number;
    labelToken?: number; // Required for CE loss
};

export async function computeGraphOptimizedStream(
    params: ComputeGraphOptimizedParams,
    onProgress?: (progress: GraphProgress) => void,
): Promise<GraphData> {
    const url = apiUrl("/api/graphs/optimized/stream");
    url.searchParams.set("prompt_id", String(params.promptId));
    url.searchParams.set("imp_min_coeff", String(params.impMinCoeff));
    url.searchParams.set("steps", String(params.steps));
    url.searchParams.set("pnorm", String(params.pnorm));
    url.searchParams.set("beta", String(params.beta));
    url.searchParams.set("normalize", String(params.normalize));
    url.searchParams.set("ci_threshold", String(params.ciThreshold));
    url.searchParams.set("mask_type", params.maskType);
    url.searchParams.set("loss_type", params.lossType);
    url.searchParams.set("loss_coeff", String(params.lossCoeff));
    url.searchParams.set("loss_position", String(params.lossPosition));
    if (params.labelToken !== undefined) {
        url.searchParams.set("label_token", String(params.labelToken));
    }

    const response = await fetch(url.toString(), { method: "POST" });
    if (!response.ok) {
        const error = await response.json();
        throw new ApiError(error.detail || `HTTP ${response.status}`, response.status);
    }

    return parseGraphSSEStream(response, onProgress);
}

export async function getGraphs(promptId: number, normalize: NormalizeType, ciThreshold: number): Promise<GraphData[]> {
    const url = apiUrl(`/api/graphs/${promptId}`);
    url.searchParams.set("normalize", normalize);
    url.searchParams.set("ci_threshold", String(ciThreshold));
    const graphs = await fetchJson<Omit<GraphData, "edgesBySource" | "edgesByTarget">[]>(url.toString());
    return graphs.map((g) => {
        // Extract all unique layer names from edges to detect architecture
        const layerNames = new Set<string>();
        for (const edge of g.edges) {
            layerNames.add(edge.src.split(":")[0]);
            layerNames.add(edge.tgt.split(":")[0]);
        }
        setArchitecture(Array.from(layerNames));

        const { edgesBySource, edgesByTarget } = buildEdgeIndexes(g.edges);
        return { ...g, edgesBySource, edgesByTarget };
    });
}

export async function tokenizeText(text: string): Promise<TokenizeResponse> {
    const url = apiUrl("/api/graphs/tokenize");
    url.searchParams.set("text", text);
    return fetchJson<TokenizeResponse>(url.toString(), { method: "POST" });
}

export async function getAllTokens(): Promise<TokenInfo[]> {
    const response = await fetchJson<{ tokens: TokenInfo[] }>("/api/graphs/tokens");
    return response.tokens;
}

export type GraphSummary = {
    id: number;
    graphType: GraphType;
    optimization: OptimizationResult | null;
};

export async function getGraphSummaries(promptId: number): Promise<GraphSummary[]> {
    return fetchJson<GraphSummary[]>(apiUrl(`/api/graphs/${promptId}/summaries`).toString());
}

export async function getGraphById(
    graphId: number,
    normalize: NormalizeType,
    ciThreshold: number,
): Promise<GraphData> {
    const url = apiUrl(`/api/graphs/by-id/${graphId}`);
    url.searchParams.set("normalize", normalize);
    url.searchParams.set("ci_threshold", String(ciThreshold));
    const g = await fetchJson<Omit<GraphData, "edgesBySource" | "edgesByTarget">>(url.toString());

    const layerNames = new Set<string>();
    for (const edge of g.edges) {
        layerNames.add(edge.src.split(":")[0]);
        layerNames.add(edge.tgt.split(":")[0]);
    }
    setArchitecture(Array.from(layerNames));

    const { edgesBySource, edgesByTarget } = buildEdgeIndexes(g.edges);
    return { ...g, edgesBySource, edgesByTarget };
}

export async function searchTokens(query: string, limit: number = 10): Promise<TokenInfo[]> {
    const url = apiUrl("/api/graphs/tokens/search");
    url.searchParams.set("q", query);
    url.searchParams.set("limit", String(limit));
    const response = await fetchJson<{ tokens: TokenInfo[] }>(url.toString());
    return response.tokens;
}
