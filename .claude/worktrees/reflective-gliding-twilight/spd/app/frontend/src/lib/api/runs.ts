/**
 * API client for /api/runs endpoints.
 */

import { API_URL } from "./index";

export type RunState = {
    id: number;
    wandb_path: string;
    config_yaml: string;
    has_prompts: boolean;
    prompt_count: number;
    context_length: number;
    backend_user: string;
};

export async function getStatus(): Promise<RunState | null> {
    const response = await fetch(`${API_URL}/api/status`);
    const data = await response.json();
    return data;
}

export async function getWhoami(): Promise<string> {
    const response = await fetch(`${API_URL}/api/whoami`);
    const data = await response.json();
    return data.user;
}

export async function loadRun(wandbRunPath: string, contextLength: number): Promise<void> {
    const url = new URL(`${API_URL}/api/runs/load`);
    url.searchParams.set("wandb_path", wandbRunPath);
    url.searchParams.set("context_length", String(contextLength));
    const response = await fetch(url.toString(), { method: "POST" });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to load run");
    }
}
