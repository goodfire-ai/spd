// place files you want to import through the `$lib` alias in this folder.

// Accept only lowercase hex/base36-ish 8-char IDs (typical for W&B)
const RUN_ID_RE = /^[a-z0-9]{8}$/;

// Full W&B run URL like:
// https://wandb.ai/<entity>/<project>/runs/<runId>
// https://wandb.ai/<entity>/<project>/runs/<runId>/overview
// https://wandb.ai/<entity>/<project>/runs/<runId>/files
// Optional trailing slash, path extensions (overview, files, etc.), and/or query string allowed.
const WAND_URL_RE = /^https:\/\/wandb\.ai\/[^/]+\/[^/]+\/runs\/([a-z0-9]{8})(?:\/[^?]*)?(?:\?.*)?$/;

// "wandb:" tagged form like:
// wandb:<entity>/<project>/<runId>
const WAND_TAG_RE = /^wandb:[^/]+\/[^/]+\/([a-z0-9]{8})$/;

// "entity/project/runId" form:
const WAND_PATH_RE = /^[^/]+\/[^/]+\/([a-z0-9]{8})$/;

export function parseWandbRunId(wandbString: string): string {
    // 1) Full URL
    const urlMatch = WAND_URL_RE.exec(wandbString);
    if (urlMatch) return urlMatch[1];

    // 2) "wandb:" tagged reference
    const tagMatch = WAND_TAG_RE.exec(wandbString);
    if (tagMatch) return tagMatch[1];

    // 3) "entity/project/runId" form
    const pathMatch = WAND_PATH_RE.exec(wandbString);
    if (pathMatch) return pathMatch[1];

    // 3) Bare run id
    if (RUN_ID_RE.test(wandbString)) return wandbString;

    // Anything else: error out
    throw new Error(
        `Invalid W&B run reference. Expected one of:
     - https://wandb.ai/<entity>/<project>/runs/<8-char id>[/path][?query]
     - wandb:<entity>/<project>/<8-char id>
     - <entity>/<project>/<8-char id>
     - <8-char id>
     Got: ${wandbString}`
    );
}
