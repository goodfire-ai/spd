/**
 * Shared API utilities and exports.
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const API_URL = (import.meta as any).env.VITE_API_URL || "http://localhost:8000";

/** Enable detailed transfer timing logs */
const TRANSFER_TIMING_ENABLED = true;

export class ApiError extends Error {
    constructor(
        message: string,
        public status: number,
    ) {
        super(message);
        this.name = "ApiError";
    }
}

/** Format bytes as human-readable */
function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)}MB`;
}

/** Log transfer timing using Resource Timing API */
function logTransferTiming(url: string, manualTotalMs: number, jsonSize: number) {
    if (!TRANSFER_TIMING_ENABLED) return;

    // Extract endpoint name from URL
    const endpoint = url.replace(API_URL, "").replace("/api/", "");

    // Use setTimeout to ensure the Resource Timing entry is available
    setTimeout(() => {
        const entries = performance.getEntriesByType("resource") as PerformanceResourceTiming[];
        const entry = entries.filter((e) => e.name.includes(url.replace(API_URL, ""))).pop();

        if (entry && entry.transferSize > 0) {
            // Resource Timing data is available (Timing-Allow-Origin header present)
            const waiting = entry.responseStart - entry.requestStart;
            const download = entry.responseEnd - entry.responseStart;
            const total = entry.responseEnd - entry.startTime;
            const transferSize = entry.transferSize;
            const decodedSize = entry.decodedBodySize;

            console.log(
                `[Transfer ${endpoint}]`,
                `total: ${total.toFixed(0)}ms`,
                `| TTFB: ${waiting.toFixed(0)}ms`,
                `| download: ${download.toFixed(0)}ms`,
                `| wire: ${formatBytes(transferSize)}`,
                `| decoded: ${formatBytes(decodedSize)}`,
            );
        } else {
            // Fallback: Resource Timing restricted, use manual measurements
            console.log(
                `[Transfer ${endpoint}]`,
                `total: ${manualTotalMs.toFixed(0)}ms`,
                `| json: ${formatBytes(jsonSize)}`,
                "(detailed timing unavailable - cross-origin)",
            );
        }
    }, 10);
}

export async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
    const shouldLogTiming =
        TRANSFER_TIMING_ENABLED &&
        (url.includes("/activation_contexts/") ||
            url.includes("/correlations/") ||
            url.includes("/dataset_attributions/") ||
            url.includes("/component_data/"));

    const startTime = shouldLogTiming ? performance.now() : 0;

    const response = await fetch(url, options);
    const text = await response.text();
    const data = JSON.parse(text);

    if (!response.ok) {
        throw new ApiError(data.detail || data.error || `HTTP ${response.status}`, response.status);
    }

    if (shouldLogTiming) {
        const totalMs = performance.now() - startTime;
        const jsonSize = new Blob([text]).size;
        logTransferTiming(url, totalMs, jsonSize);
    }

    return data as T;
}

// Re-export all API modules
export * from "./runs";
export * from "./graphs";
export * from "./prompts";
export * from "./activationContexts";
export * from "./correlations";
export * from "./datasetAttributions";
export * from "./intervention";
export * from "./dataset";
export * from "./clusters";
export * from "./componentData";
