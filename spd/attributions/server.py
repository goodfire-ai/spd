"""FastAPI server for querying the local attribution database.

Supports multiple runs in a single database. Users select a run to load,
which loads the model and enables prompt viewing.

Usage:
    python -m spd.attributions.server --db_path ./local_attr.db --port 8765

API Endpoints:
    GET  /api/runs                             - List all runs in the database
    POST /api/runs/load                        - Load a run by wandb_path
    GET  /api/status                           - Current status (loaded run, if any)
    GET  /api/activation_contexts/summary      - Component summary for loaded run
    GET  /api/activation_contexts/{layer}/{idx} - Component detail (lazy-loaded)
    GET  /api/prompts                          - List prompts for loaded run
    GET  /api/prompt/{id}                      - Prompt with on-demand graph computation
    GET  /api/prompt/{id}/optimized            - Prompt with optimized sparse CI
    GET  /api/search                           - Find prompts with specific components
"""

import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import unquote

import torch
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from spd.attributions.compute import (
    compute_local_attributions,
    compute_local_attributions_optimized,
    get_sources_by_target,
)
from spd.attributions.db import LocalAttrDB, Run
from spd.attributions.edge_normalization import normalize_edges_by_target
from spd.attributions.optim_cis.run_optim_cis import OptimCIConfig
from spd.configs import ImportanceMinimalityLossConfig, SamplingType
from spd.models.component_model import ComponentModel, SPDRunInfo

THIS_DIR = Path(__file__).parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Regex patterns for W&B path parsing
RUN_ID_RE = re.compile(r"^[a-z0-9]{8}$")
# Compact: entity/project/runid
WANDB_PATH_RE = re.compile(r"^([^/\s]+)/([^/\s]+)/([a-z0-9]{8})$")
# With /runs/: entity/project/runs/runid
WANDB_PATH_WITH_RUNS_RE = re.compile(r"^([^/\s]+)/([^/\s]+)/runs/([a-z0-9]{8})$")
# Full URL
WANDB_URL_RE = re.compile(
    r"^https://wandb\.ai/([^/]+)/([^/]+)/runs/([a-z0-9]{8})(?:/[^?]*)?(?:\?.*)?$"
)


def parse_wandb_run_path(input_str: str) -> str:
    """Parse various W&B run reference formats into normalized entity/project/runId.

    Accepts:
    - Compact form: "entity/project/xxxxxxxx"
    - With /runs/: "entity/project/runs/xxxxxxxx"
    - With wandb: prefix: "wandb:entity/project/runs/xxxxxxxx"
    - Full URL: "https://wandb.ai/entity/project/runs/xxxxxxxx[/path][?query]"

    Returns:
        Normalized path like "entity/project/xxxxxxxx"

    Raises:
        ValueError: If input doesn't match expected formats
    """
    s = input_str.strip()

    # Strip wandb: prefix if present
    if s.startswith("wandb:"):
        s = s[6:]

    # Try compact form: entity/project/runid
    m = WANDB_PATH_RE.match(s)
    if m:
        entity, project, run_id = m.groups()
        if not RUN_ID_RE.match(run_id):
            raise ValueError(f"Invalid run id: {run_id}")
        return f"{entity}/{project}/{run_id}"

    # Try with /runs/: entity/project/runs/runid
    m = WANDB_PATH_WITH_RUNS_RE.match(s)
    if m:
        entity, project, run_id = m.groups()
        if not RUN_ID_RE.match(run_id):
            raise ValueError(f"Invalid run id: {run_id}")
        return f"{entity}/{project}/{run_id}"

    # Try full URL
    m = WANDB_URL_RE.match(s)
    if m:
        entity, project, run_id = m.groups()
        if not RUN_ID_RE.match(run_id):
            raise ValueError(f"Invalid run id in URL: {run_id}")
        return f"{entity}/{project}/{run_id}"

    raise ValueError(
        f'Invalid W&B run reference. Expected either:\n'
        f' - "entity/project/xxxxxxxx" (8-char lowercase id)\n'
        f' - "wandb:entity/project/runs/xxxxxxxx"\n'
        f' - "https://wandb.ai/<entity>/<project>/runs/<8-char id>"\n'
        f"Got: {input_str}"
    )


@dataclass
class LoadedRun:
    """State for a loaded run (model, tokenizer, etc.)"""

    run: Run
    model: ComponentModel
    tokenizer: PreTrainedTokenizerFast
    sources_by_target: dict[str, list[str]]
    sampling: SamplingType
    activation_contexts_cache: dict[str, Any] | None = None


@dataclass
class AppState:
    """Server state. DB is always available; loaded_run is set after /api/runs/load."""

    db: LocalAttrDB
    loaded_run: LoadedRun | None = field(default=None)


_state: AppState | None = None


def get_state() -> AppState:
    """Get app state. Fails fast if not initialized."""
    assert _state is not None, "App state not initialized - lifespan not started"
    return _state


def get_loaded_run() -> LoadedRun:
    """Get loaded run. Fails fast if no run is loaded."""
    state = get_state()
    if state.loaded_run is None:
        raise ValueError("No run loaded. Call POST /api/runs/load first.")
    return state.loaded_run


@asynccontextmanager
async def lifespan(app: FastAPI):  # pyright: ignore[reportUnusedParameter]
    """Initialize DB connection at startup. Model loaded on-demand via /api/runs/load."""
    import os
    import sys

    global _state

    db_path_str = os.environ.get("LOCAL_ATTR_DB_PATH")
    assert db_path_str is not None, "LOCAL_ATTR_DB_PATH env var must be set"

    db = LocalAttrDB(Path(db_path_str), check_same_thread=False)
    print(f"[STARTUP] DB initialized: {db_path_str}", flush=True)
    print(f"[STARTUP] Device: {DEVICE}", flush=True)
    print(f"[STARTUP] CUDA available: {torch.cuda.is_available()}", flush=True)

    runs = db.get_all_runs()
    print(f"[STARTUP] Found {len(runs)} runs in database", flush=True)
    for run in runs:
        print(f"  - Run {run.id}: {run.wandb_path}", flush=True)
    sys.stdout.flush()

    _state = AppState(db=db)

    yield  # App runs here

    _state.db.close()


app = FastAPI(title="Local Attributions API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# -----------------------------------------------------------------------------
# Run management endpoints
# -----------------------------------------------------------------------------


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    """List all runs in the database."""
    state = get_state()
    runs = state.db.get_all_runs()
    return [
        {
            "id": run.id,
            "wandb_path": run.wandb_path,
            "n_blocks": run.n_blocks,
            "prompt_count": state.db.get_prompt_count(run.id),
        }
        for run in runs
    ]


@app.post("/api/runs/load")
def load_run(wandb_path: str):
    """Load a run by its wandb path. This loads the model onto GPU.

    Args:
        wandb_path: W&B path (entity/project/runid) or full URL
    """
    state = get_state()

    # Parse and normalize the wandb path
    try:
        normalized_path = parse_wandb_run_path(unquote(wandb_path))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # Construct full path as stored in DB: wandb:entity/project/runs/runid
    parts = normalized_path.split("/")
    full_wandb_path = f"wandb:{parts[0]}/{parts[1]}/runs/{parts[2]}"

    # Find the run in DB
    run = state.db.get_run_by_wandb_path(full_wandb_path)
    if run is None:
        return JSONResponse(
            {"error": f"Run not found in database: {full_wandb_path}"}, status_code=404
        )

    # If already loaded, skip
    if state.loaded_run is not None and state.loaded_run.run.id == run.id:
        print(f"[API] Run {run.id} already loaded, skipping")
        return {"status": "already_loaded", "run_id": run.id}

    # Unload previous run if any
    if state.loaded_run is not None:
        print(f"[API] Unloading previous run {state.loaded_run.run.id}")
        del state.loaded_run.model
        torch.cuda.empty_cache()
        state.loaded_run = None

    # Load the model
    print(f"[API] Loading run {run.id}: {run.wandb_path}")
    run_info = SPDRunInfo.from_path(run.wandb_path)
    model = ComponentModel.from_run_info(run_info)
    model = model.to(DEVICE)
    model.eval()

    # Load tokenizer
    spd_config = run_info.config
    assert spd_config.tokenizer_name is not None
    loaded_tokenizer = AutoTokenizer.from_pretrained(spd_config.tokenizer_name)
    assert isinstance(loaded_tokenizer, PreTrainedTokenizerFast)

    # Build sources_by_target mapping
    sampling = spd_config.sampling
    sources_by_target = get_sources_by_target(model, DEVICE, sampling, run.n_blocks)

    state.loaded_run = LoadedRun(
        run=run,
        model=model,
        tokenizer=loaded_tokenizer,
        sources_by_target=sources_by_target,
        sampling=sampling,
    )

    print(f"[API] Run {run.id} loaded on {DEVICE}")
    return {"status": "loaded", "run_id": run.id, "wandb_path": run.wandb_path}


@app.get("/api/status")
def get_status() -> dict[str, Any]:
    """Get current server status."""
    state = get_state()
    result: dict[str, Any] = {
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "loaded_run": None,
    }
    if state.loaded_run is not None:
        result["loaded_run"] = {
            "id": state.loaded_run.run.id,
            "wandb_path": state.loaded_run.run.wandb_path,
            "n_blocks": state.loaded_run.run.n_blocks,
            "prompt_count": state.db.get_prompt_count(state.loaded_run.run.id),
        }
    return result


# -----------------------------------------------------------------------------
# Activation contexts endpoints (require loaded run)
# -----------------------------------------------------------------------------


def _ensure_activation_contexts_cached() -> dict[str, Any] | None:
    """Load activation contexts into cache if not already loaded."""
    loaded = get_loaded_run()
    state = get_state()
    if loaded.activation_contexts_cache is None:
        contexts = state.db.get_activation_contexts(loaded.run.id)
        if contexts is not None:
            loaded.activation_contexts_cache = contexts
    return loaded.activation_contexts_cache


@app.get("/api/activation_contexts/summary")
def get_activation_contexts_summary():
    """Return lightweight summary of activation contexts (just idx + mean_ci per component)."""
    try:
        contexts = _ensure_activation_contexts_cached()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    if contexts is None:
        return JSONResponse({"error": "No activation contexts found"}, status_code=404)

    summary: dict[str, list[dict[str, Any]]] = {}
    for layer, subcomps in contexts.items():
        summary[layer] = [
            {"subcomponent_idx": s["subcomponent_idx"], "mean_ci": s["mean_ci"]}
            for s in subcomps
        ]
    return summary


@app.get("/api/activation_contexts/{layer}/{component_idx}")
def get_activation_context_detail(layer: str, component_idx: int):
    """Return full activation context data for a single component (lazy-loaded on hover/pin)."""
    try:
        contexts = _ensure_activation_contexts_cached()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    if contexts is None:
        return JSONResponse({"error": "No activation contexts found"}, status_code=404)

    layer_data = contexts.get(layer)
    if layer_data is None:
        return JSONResponse({"error": f"Layer '{layer}' not found"}, status_code=404)

    for subcomp in layer_data:
        if subcomp["subcomponent_idx"] == component_idx:
            return subcomp

    return JSONResponse(
        {"error": f"Component {component_idx} not found in layer '{layer}'"},
        status_code=404,
    )


# -----------------------------------------------------------------------------
# Custom prompt endpoints (require loaded run)
# -----------------------------------------------------------------------------


@app.post("/api/tokenize")
def tokenize_text(text: str):
    """Tokenize text and return tokens for preview.

    Args:
        text: Raw text to tokenize
    """
    try:
        loaded = get_loaded_run()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    token_ids = loaded.tokenizer.encode(text)
    token_strings = [loaded.tokenizer.decode([t]) for t in token_ids]

    return {
        "token_ids": token_ids,
        "tokens": token_strings,
        "text": text,
    }


@app.post("/api/prompt/custom")
def compute_custom_prompt(
    token_ids: list[int],
    normalize: Annotated[bool, Query()] = True,
    ci_threshold: Annotated[float, Query(ge=0)] = 1e-6,
    output_prob_threshold: Annotated[float, Query(ge=0, le=1)] = 0.01,
):
    """Compute attribution graph for custom token IDs (not stored in DB).

    Args:
        token_ids: List of token IDs to compute attributions for
        normalize: If True, normalize incoming edges to each node to sum to 1
        ci_threshold: Threshold for considering a component alive
        output_prob_threshold: Threshold for considering an output token alive
    """
    import time

    try:
        loaded = get_loaded_run()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    if not token_ids:
        return JSONResponse({"error": "No token IDs provided"}, status_code=400)

    token_strings = [loaded.tokenizer.decode([t]) for t in token_ids]
    print(
        f"[API] /api/prompt/custom called with {len(token_ids)} tokens: {token_strings}",
        flush=True,
    )

    tokens_tensor = torch.tensor([token_ids], device=DEVICE)
    t_start = time.time()
    result = compute_local_attributions(
        model=loaded.model,
        tokens=tokens_tensor,
        sources_by_target=loaded.sources_by_target,
        ci_threshold=ci_threshold,
        output_prob_threshold=output_prob_threshold,
        sampling=loaded.sampling,
        device=DEVICE,
        show_progress=True,
    )
    t_end = time.time()

    print(
        f"[API] /api/prompt/custom computed in {t_end - t_start:.2f}s, {len(result.edges)} edges",
        flush=True,
    )

    edges = result.edges
    edges.sort(key=lambda x: abs(x[6]), reverse=True)
    edges = edges[:30_000]

    if normalize:
        edges = normalize_edges_by_target(edges)

    edges_dicts = [
        {
            "src": f"{e[0]}:{e[4]}:{e[2]}",
            "tgt": f"{e[1]}:{e[5]}:{e[3]}",
            "val": e[6],
        }
        for e in edges
    ]

    output_probs: dict[str, dict[str, float | str]] = {}
    output_probs_tensor = result.output_probs[0].cpu()

    for s in range(output_probs_tensor.shape[0]):
        for c_idx in range(output_probs_tensor.shape[1]):
            prob = float(output_probs_tensor[s, c_idx].item())
            key = f"{s}:{c_idx}"
            output_probs[key] = {
                "prob": round(prob, 6),
                "token": loaded.tokenizer.decode([c_idx]),
            }

    return {
        "id": None,  # Custom prompt, not from DB
        "tokens": token_strings,
        "edges": edges_dicts,
        "outputProbs": output_probs,
    }


# -----------------------------------------------------------------------------
# Prompt endpoints (require loaded run)
# -----------------------------------------------------------------------------


@app.get("/api/prompts")
def get_prompts():
    """Return list of all prompts for the loaded run."""
    print("[API] /api/prompts called")
    state = get_state()

    try:
        loaded = get_loaded_run()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    prompt_ids = state.db.get_all_prompt_ids(loaded.run.id)
    print(f"[API] /api/prompts found {len(prompt_ids)} prompts")

    results: list[dict[str, Any]] = []
    for pid in prompt_ids:
        prompt = state.db.get_prompt(pid)
        assert prompt is not None, f"Prompt {pid} in index but not in DB"
        token_strings = [loaded.tokenizer.decode([t]) for t in prompt.token_ids]
        results.append(
            {
                "id": prompt.id,
                "tokens": token_strings,
                "preview": "".join(token_strings[:10])
                + ("..." if len(token_strings) > 10 else ""),
            }
        )
    print(f"[API] /api/prompts done: {len(results)} results")
    return results


@app.get("/api/prompt/{prompt_id}")
def get_prompt(
    prompt_id: int,
    max_mean_ci: Annotated[float, Query(ge=0, le=1)] = 1.0,
    normalize: Annotated[bool, Query()] = True,
    ci_threshold: Annotated[float, Query(ge=0)] = 1e-6,
    output_prob_threshold: Annotated[float, Query(ge=0, le=1)] = 0.01,
):
    """Return prompt data with on-demand graph computation."""
    import time

    print(
        f"[API] /api/prompt/{prompt_id} called"
        f"\n  - ci_threshold={ci_threshold}"
        f"\n  - output_prob_threshold={output_prob_threshold}"
        f"\n  - normalizing={normalize}"
        f"\n  - max_mean_ci={max_mean_ci}",
        flush=True,
    )

    state = get_state()
    try:
        loaded = get_loaded_run()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    prompt = state.db.get_prompt(prompt_id)
    if prompt is None:
        return JSONResponse({"error": f"Prompt {prompt_id} not found"}, status_code=404)

    # Verify prompt belongs to loaded run
    if prompt.run_id != loaded.run.id:
        return JSONResponse(
            {"error": f"Prompt {prompt_id} belongs to run {prompt.run_id}, not loaded run {loaded.run.id}"},
            status_code=400,
        )

    token_ids = prompt.token_ids
    token_strings = [loaded.tokenizer.decode([t]) for t in token_ids]
    print(
        f"[API] /api/prompt/{prompt_id} computing attributions for {len(token_strings)} tokens on device={DEVICE}...",
        flush=True,
    )

    tokens_tensor = torch.tensor([token_ids], device=DEVICE)
    t_compute_start = time.time()
    result = compute_local_attributions(
        model=loaded.model,
        tokens=tokens_tensor,
        sources_by_target=loaded.sources_by_target,
        ci_threshold=ci_threshold,
        output_prob_threshold=output_prob_threshold,
        sampling=loaded.sampling,
        device=DEVICE,
        show_progress=True,
    )
    t_compute_end = time.time()

    print(
        f"[API] /api/prompt/{prompt_id} graph computed in {t_compute_end - t_compute_start:.2f}s, {len(result.edges)} edges",
        flush=True,
    )

    # Build mean CI lookup for filtering
    mean_ci_lookup: dict[str, float] = {}
    if max_mean_ci < 1.0:
        activation_contexts = _ensure_activation_contexts_cached()
        if activation_contexts:
            for layer, subcomps in activation_contexts.items():
                for subcomp in subcomps:
                    mean_ci_lookup[f"{layer}:{subcomp['subcomponent_idx']}"] = subcomp[
                        "mean_ci"
                    ]

    edges = result.edges

    if max_mean_ci < 1.0 and mean_ci_lookup:
        edges = [
            e
            for e in edges
            if mean_ci_lookup.get(f"{e[0]}:{e[2]}", 0.0) <= max_mean_ci
            and mean_ci_lookup.get(f"{e[1]}:{e[3]}", 0.0) <= max_mean_ci
        ]
        print(
            f"[API] /api/prompt/{prompt_id} filtered to {len(edges)} edges by max_mean_ci={max_mean_ci}"
        )

    edges.sort(key=lambda x: abs(x[6]), reverse=True)
    edges = edges[:30_000]

    if normalize:
        edges = normalize_edges_by_target(edges)

    edges_dicts = [
        {
            "src": f"{e[0]}:{e[4]}:{e[2]}",
            "tgt": f"{e[1]}:{e[5]}:{e[3]}",
            "val": e[6],
        }
        for e in edges
    ]

    output_probs: dict[str, dict[str, float | str]] = {}
    output_probs_tensor = result.output_probs[0].cpu()

    for s in range(output_probs_tensor.shape[0]):
        for c_idx in range(output_probs_tensor.shape[1]):
            prob = float(output_probs_tensor[s, c_idx].item())
            key = f"{s}:{c_idx}"
            output_probs[key] = {
                "prob": round(prob, 6),
                "token": loaded.tokenizer.decode([c_idx]),
            }

    return {
        "id": prompt.id,
        "tokens": token_strings,
        "edges": edges_dicts,
        "outputProbs": output_probs,
    }


@app.get("/api/prompt/{prompt_id}/optimized")
def get_prompt_optimized(
    prompt_id: int,
    label_token: Annotated[int | None, Query()] = None,
    imp_min_coeff: Annotated[float, Query(gt=0)] = 0.1,
    ce_loss_coeff: Annotated[float, Query(gt=0)] = 1.0,
    steps: Annotated[int, Query(gt=0)] = 500,
    lr: Annotated[float, Query(gt=0)] = 1e-2,
    pnorm: Annotated[float, Query(gt=0, le=1)] = 0.3,
    normalize: Annotated[bool, Query()] = True,
    ci_threshold: Annotated[float, Query(ge=0)] = 1e-6,
    output_prob_threshold: Annotated[float, Query(ge=0, le=1)] = 0.01,
):
    """Return prompt data with optimized sparse CI values."""
    import time

    state = get_state()
    try:
        loaded = get_loaded_run()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    prompt = state.db.get_prompt(prompt_id)
    if prompt is None:
        return JSONResponse({"error": f"Prompt {prompt_id} not found"}, status_code=404)

    if prompt.run_id != loaded.run.id:
        return JSONResponse(
            {"error": f"Prompt {prompt_id} belongs to run {prompt.run_id}, not loaded run {loaded.run.id}"},
            status_code=400,
        )

    token_ids = prompt.token_ids
    tokens_tensor = torch.tensor([token_ids], device=DEVICE)

    if label_token is None:
        with torch.no_grad():
            logits = loaded.model(tokens_tensor)
            label_token = int(logits[0, -1, :].argmax().item())

    label_str = loaded.tokenizer.decode([label_token])
    token_strings = [loaded.tokenizer.decode([t]) for t in token_ids]

    print(
        f"[API] /api/prompt/{prompt_id}/optimized called"
        f"\n  - label_token={label_token} ({label_str!r})"
        f"\n  - imp_min_coeff={imp_min_coeff}, ce_loss_coeff={ce_loss_coeff}"
        f"\n  - steps={steps}, lr={lr}, pnorm={pnorm}"
        f"\n  - tokens: {token_strings}",
        flush=True,
    )

    optim_config = OptimCIConfig(
        seed=0,
        lr=lr,
        steps=steps,
        weight_decay=0.0,
        lr_schedule="cosine",
        lr_exponential_halflife=None,
        lr_warmup_pct=0.01,
        log_freq=max(1, steps // 4),
        imp_min_config=ImportanceMinimalityLossConfig(coeff=imp_min_coeff, pnorm=pnorm),
        ce_loss_coeff=ce_loss_coeff,
        ci_threshold=ci_threshold,
        sampling=loaded.sampling,
        ce_kl_rounding_threshold=0.5,
    )

    t_start = time.time()
    result = compute_local_attributions_optimized(
        model=loaded.model,
        tokens=tokens_tensor,
        label_token=label_token,
        sources_by_target=loaded.sources_by_target,
        optim_config=optim_config,
        ci_threshold=ci_threshold,
        output_prob_threshold=output_prob_threshold,
        device=DEVICE,
        show_progress=True,
    )
    t_end = time.time()

    print(
        f"[API] /api/prompt/{prompt_id}/optimized completed in {t_end - t_start:.2f}s, "
        f"{len(result.edges)} edges",
        flush=True,
    )

    edges = result.edges
    edges.sort(key=lambda x: abs(x[6]), reverse=True)
    edges = edges[:30_000]

    if normalize:
        edges = normalize_edges_by_target(edges)

    edges_dicts = [
        {
            "src": f"{e[0]}:{e[4]}:{e[2]}",
            "tgt": f"{e[1]}:{e[5]}:{e[3]}",
            "val": e[6],
        }
        for e in edges
    ]

    output_probs: dict[str, dict[str, float | str]] = {}
    output_probs_tensor = result.output_probs[0].cpu()
    for s in range(output_probs_tensor.shape[0]):
        for c_idx in range(output_probs_tensor.shape[1]):
            prob = float(output_probs_tensor[s, c_idx].item())
            key = f"{s}:{c_idx}"
            output_probs[key] = {
                "prob": round(prob, 6),
                "token": loaded.tokenizer.decode([c_idx]),
            }

    return {
        "id": prompt.id,
        "tokens": token_strings,
        "edges": edges_dicts,
        "outputProbs": output_probs,
        "optimization": {
            "label_token": label_token,
            "label_str": label_str,
            "imp_min_coeff": imp_min_coeff,
            "ce_loss_coeff": ce_loss_coeff,
            "steps": steps,
            "label_prob": result.stats.label_prob,
            "l0_total": result.stats.l0_total,
            "l0_per_layer": result.stats.l0_per_layer,
        },
    }


@app.get("/api/search")
def search_prompts(
    components: str = "",
    mode: Annotated[str, Query(pattern="^(all|any)$")] = "all",
):
    """Search for prompts with specified components in the loaded run."""
    state = get_state()
    try:
        loaded = get_loaded_run()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    component_list = [c.strip() for c in components.split(",") if c.strip()]
    if not component_list:
        return JSONResponse({"error": "No components specified"}, status_code=400)

    require_all = mode == "all"
    prompt_ids = state.db.find_prompts_with_components(
        loaded.run.id, component_list, require_all=require_all
    )

    results: list[dict[str, Any]] = []
    for pid in prompt_ids:
        prompt = state.db.get_prompt(pid)
        assert prompt is not None, f"Prompt {pid} in index but not in DB"
        token_strings = [loaded.tokenizer.decode([t]) for t in prompt.token_ids]
        results.append(
            {
                "id": prompt.id,
                "tokens": token_strings,
                "preview": "".join(token_strings[:10])
                + ("..." if len(token_strings) > 10 else ""),
            }
        )

    return {
        "query": {"components": component_list, "mode": mode},
        "count": len(results),
        "results": results,
    }


# -----------------------------------------------------------------------------
# Static files
# -----------------------------------------------------------------------------


@app.get("/")
def index():
    """Serve the main HTML file."""
    return FileResponse(THIS_DIR / "local_attributions_alpine.html")


@app.get("/{filename:path}")
def static_file(filename: str):
    """Serve static files from the attributions directory."""
    file_path = THIS_DIR / filename
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    return JSONResponse({"error": "Not found"}, status_code=404)


# -----------------------------------------------------------------------------
# Server startup
# -----------------------------------------------------------------------------


def main(db_path: str, port: int = 8765, workers: int = 1):
    """Run the server.

    Args:
        db_path: Path to SQLite database
        port: Port to serve on (default 8765)
        workers: Number of uvicorn workers. Each loads the model, so 1 is usually best.
    """
    import os
    import signal

    import uvicorn

    db_path_ = Path(db_path)
    assert db_path_.exists(), f"Database not found: {db_path_}"

    os.environ["LOCAL_ATTR_DB_PATH"] = str(db_path_)

    def force_exit(
        sig,  # pyright: ignore[reportUnusedParameter, reportUnknownParameterType, reportMissingParameterType]
        frame,  # pyright: ignore[reportUnusedParameter, reportUnknownParameterType, reportMissingParameterType]
    ):
        print("\nForce exiting...")
        os._exit(1)

    signal.signal(signal.SIGINT, force_exit)
    signal.signal(signal.SIGTERM, force_exit)

    print(f"Server running at http://localhost:{port}/")

    uvicorn.run(
        "spd.attributions.server:app",
        host="localhost",
        port=port,
        log_level="warning",
        workers=workers,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
