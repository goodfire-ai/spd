/**
 * Registry of canonical SPD runs for quick access in the app.
 */

export type RegistryEntry = {
    /** Full wandb run id (e.g., "goodfire/spd/jyo9duz5") */
    wandbRunId: string;
    /** Human-readable model name */
    modelName: string;
    /** Optional notes about the run */
    notes?: string;
};

const DEFAULT_ENTITY_PROJECT = "goodfire/spd";

/**
 * Canonical runs registry - add new entries here.
 * These appear in the dropdown for quick selection.
 */
export const CANONICAL_RUNS: RegistryEntry[] = [
    {
        wandbRunId: "goodfire/spd/vjbol27n",
        modelName: "ss_llama_simple_mlp-1.25M (4L)",
        notes: "Lucius' run, Dec 8",
    },
    {
        wandbRunId: "goodfire/spd/278we8gk",
        modelName: "ss_llama_simple_mlp-1.25M (4L)",
        notes: "Dan's initial run, Dec 6",
    },
    {
        wandbRunId: "goodfire/spd/jyo9duz5",
        modelName: "ss_gpt2_simple-1.25M (4L)",
    },
    {
        wandbRunId: "goodfire/spd/5cr21lbs",
        modelName: "ss_llama_simple_mlp (1L)",
    },
    {
        wandbRunId: "goodfire/spd/itmexlj0",
        modelName: "ss_llama_simple_mlp (2L)",
    },
    {
        wandbRunId: "goodfire/spd/33n6xjjt",
        modelName: "ss_gpt2_simple (1L)",
    },
];

/**
 * Formats a wandb run id for display.
 * Shows just the 8-char run id if it's from "goodfire/spd",
 * otherwise shows the full path.
 */
export function formatRunIdForDisplay(wandbRunId: string): string {
    if (wandbRunId.startsWith(`${DEFAULT_ENTITY_PROJECT}/`)) {
        // Extract just the run id (last segment)
        const parts = wandbRunId.split("/");
        return parts[parts.length - 1];
    }
    return wandbRunId;
}
