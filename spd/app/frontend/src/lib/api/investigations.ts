/**
 * API client for investigations (agent swarm results).
 * Investigations are flattened across swarms - each task is independent.
 */

export interface InvestigationSummary {
    id: string; // swarm_id/task_id
    swarm_id: string;
    task_id: number;
    wandb_path: string | null;
    created_at: string;
    has_research_log: boolean;
    has_explanations: boolean;
    event_count: number;
    last_event_time: string | null;
    last_event_message: string | null;
    // Agent-provided summary
    title: string | null;
    summary: string | null;
    status: string | null; // in_progress, completed, inconclusive
}

export interface EventEntry {
    event_type: string;
    timestamp: string;
    message: string;
    details: Record<string, unknown> | null;
}

export interface InvestigationDetail {
    id: string;
    swarm_id: string;
    task_id: number;
    wandb_path: string | null;
    created_at: string;
    research_log: string | null;
    events: EventEntry[];
    explanations: Record<string, unknown>[];
    // Agent-provided summary
    title: string | null;
    summary: string | null;
    status: string | null;
}

export async function listInvestigations(): Promise<InvestigationSummary[]> {
    const res = await fetch("/api/investigations");
    if (!res.ok) throw new Error(`Failed to list investigations: ${res.statusText}`);
    return res.json();
}

export async function getInvestigation(swarmId: string, taskId: number): Promise<InvestigationDetail> {
    const res = await fetch(`/api/investigations/${swarmId}/${taskId}`);
    if (!res.ok) throw new Error(`Failed to get investigation: ${res.statusText}`);
    return res.json();
}
