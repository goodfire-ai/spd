/**
 * API client for /api/prompts endpoints.
 */

import type { PromptPreview } from "../promptAttributionsTypes";
import { API_URL, fetchJson } from "./index";

export async function listPrompts(): Promise<PromptPreview[]> {
    return fetchJson<PromptPreview[]>(`${API_URL}/api/prompts`);
}

export async function createCustomPrompt(text: string): Promise<PromptPreview> {
    const url = new URL(`${API_URL}/api/prompts/custom`);
    url.searchParams.set("text", text);
    return fetchJson<PromptPreview>(url.toString(), { method: "POST" });
}
