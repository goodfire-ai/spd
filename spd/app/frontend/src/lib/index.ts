// Accept only lowercase hex/base36-ish 8-char IDs (typical for W&B)
const RUN_ID_RE = /^[a-z0-9]{8}$/;

// Compact form: entity/project/runId
const WAND_PATH_RE = /^([^/\s]+)\/([^/\s]+)\/([a-z0-9]{8})$/;

// Full W&B run URL like:
// https://wandb.ai/<entity>/<project>/runs/<runId>[/path][?query]
const WAND_URL_RE = /^https:\/\/wandb\.ai\/([^/]+)\/([^/]+)\/runs\/([a-z0-9]{8})(?:\/[^?]*)?(?:\?.*)?$/;

export function parseWandbRunPath(input: string): string {
    const s = input.trim();

    // 1) Try compact form first: entity/project/runid
    let m = WAND_PATH_RE.exec(s);
    if (m) {
        const [, entity, project, runId] = m;
        if (!RUN_ID_RE.test(runId)) {
            throw new Error(`Invalid run id: ${runId}`);
        }
        return `${entity}/${project}/${runId}`;
    }

    // 2) Try full URL
    m = WAND_URL_RE.exec(s);
    if (m) {
        const [, entity, project, runId] = m;
        if (!RUN_ID_RE.test(runId)) {
            throw new Error(`Invalid run id in URL: ${runId}`);
        }
        return `${entity}/${project}/${runId}`;
    }

    // 3) Fail with a helpful message
    throw new Error(
        `Invalid W&B run reference. Expected either:
 - "entity/project/xxxxxxxx" (8-char lowercase id)
 - "https://wandb.ai/<entity>/<project>/runs/<8-char id>[/path][?query]"
Got: ${input}`,
    );
}
