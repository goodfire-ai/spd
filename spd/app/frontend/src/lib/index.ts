// Accept only lowercase hex/base36-ish 8-char IDs (typical for W&B)
const RUN_ID_RE = /^[a-z0-9]{8}$/;

// Full W&B run URL like:
// https://wandb.ai/<entity>/<project>/runs/<runId>[/path][?query]
const WAND_URL_RE =
  /^https:\/\/wandb\.ai\/([^/]+)\/([^/]+)\/runs\/([a-z0-9]{8})(?:\/[^?]*)?(?:\?.*)?$/;

export function parseWandbRunId(wandbUrl: string): string {
  const m = WAND_URL_RE.exec(wandbUrl);
  if (!m) {
    throw new Error(
      `Invalid W&B run URL. Expected:
 - https://wandb.ai/<entity>/<project>/runs/<8-char id>[/path][?query]
Got: ${wandbUrl}`
    );
  }

  const [, entity, project, runId] = m;

  // (Optional) extra safety: ensure runId matches our expected pattern
  if (!RUN_ID_RE.test(runId)) {
    throw new Error(`Invalid run id in URL: ${runId}`);
  }

  return `${entity}/${project}/${runId}`;
}
