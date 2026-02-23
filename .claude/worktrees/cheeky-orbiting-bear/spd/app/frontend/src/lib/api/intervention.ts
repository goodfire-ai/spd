/**
 * API client for /api/intervention endpoints.
 */

import type { ForkedInterventionRun, InterventionRunSummary, RunInterventionRequest } from "../interventionTypes";
import { API_URL } from "./index";

export async function runAndSaveIntervention(request: RunInterventionRequest): Promise<InterventionRunSummary> {
    const response = await fetch(`${API_URL}/api/intervention/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to run intervention");
    }
    return (await response.json()) as InterventionRunSummary;
}

export async function getInterventionRuns(graphId: number): Promise<InterventionRunSummary[]> {
    const response = await fetch(`${API_URL}/api/intervention/runs/${graphId}`);
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to get intervention runs");
    }
    return (await response.json()) as InterventionRunSummary[];
}

export async function deleteInterventionRun(runId: number): Promise<void> {
    const response = await fetch(`${API_URL}/api/intervention/runs/${runId}`, {
        method: "DELETE",
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to delete intervention run");
    }
}

export async function forkInterventionRun(
    runId: number,
    tokenReplacements: [number, number][],
    topK: number = 10,
): Promise<ForkedInterventionRun> {
    const response = await fetch(`${API_URL}/api/intervention/runs/${runId}/fork`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token_replacements: tokenReplacements, top_k: topK }),
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to fork intervention run");
    }
    return (await response.json()) as ForkedInterventionRun;
}

export async function deleteForkedInterventionRun(forkId: number): Promise<void> {
    const response = await fetch(`${API_URL}/api/intervention/forks/${forkId}`, {
        method: "DELETE",
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to delete forked intervention run");
    }
}
