// Accept only lowercase hex/base36-ish 8-char IDs (typical for W&B)
const RUN_ID_RE = /^[a-z0-9]{8}$/;

// Compact form: entity/project/runId
const WANDB_PATH_RE = /^([^/\s]+)\/([^/\s]+)\/([a-z0-9]{8})$/;

// Form with /runs/: entity/project/runs/runId
const WANDB_PATH_WITH_RUNS_RE = /^([^/\s]+)\/([^/\s]+)\/runs\/([a-z0-9]{8})$/;

// Full W&B run URL like:
// https://wandb.ai/<entity>/<project>/runs/<runId>[/path][?query]
const WANDB_URL_RE =
    /^https:\/\/wandb\.ai\/([^/]+)\/([^/]+)\/runs\/([a-z0-9]{8})(?:\/[^?]*)?(?:\?.*)?$/;

/**
 * Parse various W&B run reference formats into normalized entity/project/runId.
 *
 * Accepts:
 * - "entity/project/runId" (compact form)
 * - "entity/project/runs/runId" (with /runs/)
 * - "wandb:entity/project/runId" (with wandb: prefix)
 * - "wandb:entity/project/runs/runId" (full wandb: form)
 * - "https://wandb.ai/entity/project/runs/runId..." (URL)
 *
 * Returns normalized form: "entity/project/runId"
 */
export function parseWandbRunPath(input: string): string {
    let s = input.trim();

    // Strip wandb: prefix if present
    if (s.startsWith("wandb:")) {
        s = s.slice(6);
    }

    // 1) Try compact form: entity/project/runid
    let m = WANDB_PATH_RE.exec(s);
    if (m) {
        const [, entity, project, runId] = m;
        if (!RUN_ID_RE.test(runId)) {
            throw new Error(`Invalid run id: ${runId}`);
        }
        return `${entity}/${project}/${runId}`;
    }

    // 2) Try form with /runs/: entity/project/runs/runid
    m = WANDB_PATH_WITH_RUNS_RE.exec(s);
    if (m) {
        const [, entity, project, runId] = m;
        if (!RUN_ID_RE.test(runId)) {
            throw new Error(`Invalid run id: ${runId}`);
        }
        return `${entity}/${project}/${runId}`;
    }

    // 3) Try full URL
    m = WANDB_URL_RE.exec(s);
    if (m) {
        const [, entity, project, runId] = m;
        if (!RUN_ID_RE.test(runId)) {
            throw new Error(`Invalid run id in URL: ${runId}`);
        }
        return `${entity}/${project}/${runId}`;
    }

    // 4) Fail with a helpful message
    throw new Error(
        `Invalid W&B run reference. Expected one of:
 - "entity/project/xxxxxxxx"
 - "entity/project/runs/xxxxxxxx"
 - "wandb:entity/project/runs/xxxxxxxx"
 - "https://wandb.ai/entity/project/runs/xxxxxxxx"
Got: "${input}"`,
    );
}
