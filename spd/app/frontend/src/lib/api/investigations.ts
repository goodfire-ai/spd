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
    artifact_ids: string[]; // List of artifact IDs available for this investigation
    // Agent-provided summary
    title: string | null;
    summary: string | null;
    status: string | null;
}

export interface GraphArtifact {
    type: "graph";
    id: string;
    caption: string | null;
    graph_id: number;
    data: {
        tokens: string[];
        edges: { src: string; tgt: string; val: number }[];
        outputProbs: Record<
            string,
            { prob: number; logit: number; target_prob: number; target_logit: number; token: string }
        >;
        nodeCiVals: Record<string, number>;
        nodeSubcompActs: Record<string, number>;
        maxAbsAttr: number;
        l0_total: number;
    };
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

export async function listArtifacts(swarmId: string, taskId: number): Promise<string[]> {
    const res = await fetch(`/api/investigations/${swarmId}/${taskId}/artifacts`);
    if (!res.ok) throw new Error(`Failed to list artifacts: ${res.statusText}`);
    return res.json();
}

export async function getArtifact(swarmId: string, taskId: number, artifactId: string): Promise<GraphArtifact> {
    const res = await fetch(`/api/investigations/${swarmId}/${taskId}/artifacts/${artifactId}`);
    if (!res.ok) throw new Error(`Failed to get artifact: ${res.statusText}`);
    return res.json();
}
