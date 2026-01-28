/**
 * Shared API utilities and exports.
 *
 * In development, Vite proxies /api requests to the backend.
 * This allows the frontend to work regardless of which port the backend is on.
 */

/**
 * Build a URL for an API endpoint.
 * Uses relative paths which Vite's proxy forwards to the backend.
 */
export function apiUrl(path: string): URL {
    return new URL(path, window.location.origin);
}

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
    const text = await response.text();
    const data = JSON.parse(text);

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
export * from "./componentData";
