/**
 * API client for /api/prompts endpoints.
 */

import type { PromptPreview } from "../promptAttributionsTypes";
import { API_URL, ApiError, fetchJson } from "./index";

export async function listPrompts(): Promise<PromptPreview[]> {
    return fetchJson<PromptPreview[]>(`${API_URL}/api/prompts`);
}

export async function createCustomPrompt(text: string): Promise<PromptPreview> {
    const url = new URL(`${API_URL}/api/prompts/custom`);
    url.searchParams.set("text", text);
    return fetchJson<PromptPreview>(url.toString(), { method: "POST" });
}

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
        throw new ApiError(error.detail || `HTTP ${response.status}`, response.status);
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
