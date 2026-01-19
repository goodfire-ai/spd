/**
 * Shared API utilities and exports.
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const API_URL = (import.meta as any).env.VITE_API_URL || "http://localhost:8000";

export class ApiError extends Error {
    constructor(
        message: string,
        public status: number,
    ) {
        super(message);
        this.name = "ApiError";
    }
}

export async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
    const response = await fetch(url, options);
    const data = await response.json();

    if (!response.ok) {
        throw new ApiError(data.detail || data.error || `HTTP ${response.status}`, response.status);
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
