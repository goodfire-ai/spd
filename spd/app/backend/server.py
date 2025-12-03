"""Unified FastAPI server for the SPD app.

Merges the main app backend with the local attributions server.
Supports multiple runs, on-demand attribution graph computation,
and activation contexts generation.

Usage:
    python -m spd.app.backend.server --port 8000
"""

import functools
import json
import re
import traceback
import uuid
from collections.abc import Callable, Generator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any
from urllib.parse import unquote

import fire
import torch
import uvicorn
import yaml
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from spd.app.backend.compute import (
    compute_ci_only,
    compute_local_attributions,
    compute_local_attributions_optimized,
    extract_active_from_ci,
    get_sources_by_target,
)
from spd.app.backend.db import LocalAttrDB, Run
from spd.app.backend.lib.activation_contexts import get_activations_data_streaming
from spd.app.backend.lib.edge_normalization import normalize_edges_by_target
from spd.app.backend.optim_cis.run_optim_cis import OptimCIConfig
from spd.app.backend.schemas import (
    ActivationContextsGenerationConfig,
    HarvestMetadata,
    LoadedRun,
    ModelActivationContexts,
    RunInfo,
    SubcomponentMetadata,
)
from spd.configs import Config, ImportanceMinimalityLossConfig, SamplingType
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import runtime_cast

DEVICE = get_device()
GLOBAL_EDGE_LIMIT = 5_000

def log_errors[T: Callable[..., Any]](func: T) -> T:
    """Decorator to log errors with full traceback for easier debugging."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except HTTPException:
            raise  # Let FastAPI handle HTTP exceptions normally
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            traceback.print_exc()
            raise

    return wrapper  # pyright: ignore[reportReturnType]


# Regex patterns for W&B path parsing
RUN_ID_RE = re.compile(r"^[a-z0-9]{8}$")
WANDB_PATH_RE = re.compile(r"^([^/\s]+)/([^/\s]+)/([a-z0-9]{8})$")
WANDB_PATH_WITH_RUNS_RE = re.compile(r"^([^/\s]+)/([^/\s]+)/runs/([a-z0-9]{8})$")
WANDB_URL_RE = re.compile(
    r"^https://wandb\.ai/([^/]+)/([^/]+)/runs/([a-z0-9]{8})(?:/[^?]*)?(?:\?.*)?$"
)


def parse_wandb_run_path(input_str: str) -> str:
    """Parse various W&B run reference formats into normalized entity/project/runId."""
    s = input_str.strip()
    if s.startswith("wandb:"):
        s = s[6:]

    m = WANDB_PATH_RE.match(s)
    if m:
        entity, project, run_id = m.groups()
        if not RUN_ID_RE.match(run_id):
            raise ValueError(f"Invalid run id: {run_id}")
        return f"{entity}/{project}/{run_id}"

    m = WANDB_PATH_WITH_RUNS_RE.match(s)
    if m:
        entity, project, run_id = m.groups()
        if not RUN_ID_RE.match(run_id):
            raise ValueError(f"Invalid run id: {run_id}")
        return f"{entity}/{project}/{run_id}"

    m = WANDB_URL_RE.match(s)
    if m:
        entity, project, run_id = m.groups()
        if not RUN_ID_RE.match(run_id):
            raise ValueError(f"Invalid run id in URL: {run_id}")
        return f"{entity}/{project}/{run_id}"

    raise ValueError(
        f"Invalid W&B run reference. Expected either:\n"
        f' - "entity/project/xxxxxxxx" (8-char lowercase id)\n'
        f' - "wandb:entity/project/runs/xxxxxxxx"\n'
        f' - "https://wandb.ai/<entity>/<project>/runs/<8-char id>"\n'
        f"Got: {input_str}"
    )


@dataclass
class RunState:
    """Runtime state for a loaded run (model, tokenizer, etc.)"""

    run: Run
    model: ComponentModel
    tokenizer: PreTrainedTokenizerBase
    sources_by_target: dict[str, list[str]]
    sampling: SamplingType
    config: Config
    token_strings: dict[int, str]
    activation_contexts_cache: dict[str, Any] | None = None


@dataclass
class AppState:
    """Server state. DB is always available; loaded_run is set after /api/runs/load."""

    db: LocalAttrDB
    run_state: RunState | None = field(default=None)


_state: AppState | None = None

# Cache for harvest results (streaming activation contexts), keyed by UUID
harvest_cache: dict[str, ModelActivationContexts] = {}


def get_state() -> AppState:
    """Get app state. Fails fast if not initialized."""
    assert _state is not None, "App state not initialized - lifespan not started"
    return _state


def get_loaded_run() -> RunState:
    """Get loaded run. Fails fast if no run is loaded."""
    state = get_state()
    if state.run_state is None:
        raise ValueError("No run loaded. Call POST /api/runs/load first.")
    return state.run_state


@asynccontextmanager
async def lifespan(app: FastAPI):  # pyright: ignore[reportUnusedParameter]
    """Initialize DB connection at startup. Model loaded on-demand via /api/runs/load."""
    global _state

    db = LocalAttrDB(check_same_thread=False)
    db.init_schema()

    logger.info(f"[STARTUP] DB initialized: {db.db_path}")
    logger.info(f"[STARTUP] Device: {DEVICE}")
    logger.info(f"[STARTUP] CUDA available: {torch.cuda.is_available()}")

    runs = db.get_all_runs()
    logger.info(f"[STARTUP] Found {len(runs)} runs in database")
    for run in runs:
        logger.info(f"  - Run {run.id}: {run.wandb_path}")

    _state = AppState(db=db)

    yield

    _state.db.close()


app = FastAPI(title="SPD App API", lifespan=lifespan, debug=True)
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
@log_errors
def list_runs() -> list[RunInfo]:
    """List all runs in the database."""
    state = get_state()
    runs = state.db.get_all_runs()
    return [
        RunInfo(
            id=run.id,
            wandb_path=run.wandb_path,
            prompt_count=state.db.get_prompt_count(run.id),
            has_activation_contexts=state.db.has_activation_contexts(run.id),
        )
        for run in runs
    ]


# Characters that don't get a space prefix in wordpiece

_PUNCT_NO_SPACE = set(".,!?;:'\")-]}>/")


def _build_token_lookup(
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_name: str,
) -> dict[int, str]:
    """Build token ID -> string lookup.

    Uses tokenizer-specific strategy to produce strings that concatenate correctly.
    """
    lookup: dict[int, str] = {}
    vocab_size: int = tokenizer.vocab_size  # pyright: ignore[reportAssignmentType]

    for tid in range(vocab_size):
        decoded: str = tokenizer.decode([tid], skip_special_tokens=False)

        # Tokenizer name -> decode strategy
        # "wordpiece": ## = continuation (strip ##), punctuation = no space, others = space prefix
        # "bpe": spaces encoded in token via Ġ, just decode directly
        match tokenizer_name:
            case "SimpleStories/test-SimpleStories-gpt2-1.25M":
                # WordPiece handling:
                if decoded.startswith("##"):
                    # Continuation token - strip ## prefix, no space
                    lookup[tid] = decoded[2:]
                elif decoded and decoded[0] in _PUNCT_NO_SPACE:
                    # Punctuation - no space prefix
                    lookup[tid] = decoded
                else:
                    # Regular token - add space prefix
                    lookup[tid] = " " + decoded
            case "openai-community/gpt2":
                # BPE (GPT-2 style): spaces encoded in token via Ġ -> space
                lookup[tid] = decoded
            case _:
                raise ValueError(f"Unsupported tokenizer name: {tokenizer_name}")

    return lookup


@app.post("/api/runs/load")
@log_errors
def load_run(wandb_path: str):
    """Load a run by its wandb path. Creates the run in DB if not found.

    This loads the model onto GPU and makes it available for attribution computation.
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

    # Load the model from W&B to get config info
    logger.info(f"[API] Loading run info from W&B: {full_wandb_path}")
    try:
        run_info = SPDRunInfo.from_path(full_wandb_path)
    except Exception as e:
        return JSONResponse({"error": f"Failed to load run from W&B: {e}"}, status_code=400)

    run = state.db.get_run_by_wandb_path(full_wandb_path)
    if run is None:
        run_id = state.db.create_run(full_wandb_path)
        run = state.db.get_run(run_id)
        assert run is not None
        logger.info(f"[API] Created new run in DB: {run.id}")
    else:
        logger.info(f"[API] Found existing run in DB: {run.id}")

    # If already loaded, skip model load
    if state.run_state is not None and state.run_state.run.id == run.id:
        logger.info(f"[API] Run {run.id} already loaded, skipping")
        return {"status": "already_loaded", "run_id": run.id, "wandb_path": run.wandb_path}

    # Unload previous run if any
    if state.run_state is not None:
        logger.info(f"[API] Unloading previous run {state.run_state.run.id}")
        del state.run_state.model
        torch.cuda.empty_cache()
        state.run_state = None

    # Load the model
    logger.info(f"[API] Loading model for run {run.id}: {run.wandb_path}")
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
    sources_by_target = get_sources_by_target(model, DEVICE, sampling)

    # Build token lookup for activation contexts
    token_strings = _build_token_lookup(loaded_tokenizer, spd_config.tokenizer_name)

    state.run_state = RunState(
        run=run,
        model=model,
        tokenizer=loaded_tokenizer,
        sources_by_target=sources_by_target,
        sampling=sampling,
        config=spd_config,
        token_strings=token_strings,
    )

    logger.info(f"[API] Run {run.id} loaded on {DEVICE}")
    return {"status": "loaded", "run_id": run.id, "wandb_path": run.wandb_path}


@app.get("/api/status")
@log_errors
def get_status() -> LoadedRun | None:
    """Get current server status."""
    state = get_state()

    if state.run_state is None:
        return None

    run = state.run_state.run
    config_yaml = yaml.dump(
        state.run_state.config.model_dump(), default_flow_style=False, sort_keys=False
    )

    return LoadedRun(
        id=run.id,
        wandb_path=run.wandb_path,
        config_yaml=config_yaml,
        has_activation_contexts=state.db.has_activation_contexts(run.id),
        has_prompts=state.db.has_prompts(run.id),
        prompt_count=state.db.get_prompt_count(run.id),
    )


# -----------------------------------------------------------------------------
# Activation contexts endpoints
# -----------------------------------------------------------------------------


def _ensure_activation_contexts_cached() -> dict[str, Any] | None:
    """Load activation contexts into cache if not already loaded."""
    loaded = get_loaded_run()
    state = get_state()
    if loaded.activation_contexts_cache is None:
        contexts = state.db.get_activation_contexts_raw(loaded.run.id)
        if contexts is not None:
            loaded.activation_contexts_cache = contexts
    return loaded.activation_contexts_cache


@app.get("/api/activation_contexts/summary")
@log_errors
def get_activation_contexts_summary():
    """Return lightweight summary of activation contexts (just idx + mean_ci per component)."""
    try:
        contexts = _ensure_activation_contexts_cached()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    if contexts is None:
        return JSONResponse(
            {"error": "No activation contexts found. Generate them first.", "missing": True},
            status_code=404,
        )

    # Raw dict has {"layers": {...}} structure
    layers = contexts.get("layers", contexts)

    summary: dict[str, list[dict[str, Any]]] = {}
    for layer, subcomps in layers.items():
        summary[layer] = [
            {"subcomponent_idx": s["subcomponent_idx"], "mean_ci": s["mean_ci"]} for s in subcomps
        ]
    return summary


@app.get("/api/activation_contexts/{layer}/{component_idx}")
@log_errors
def get_activation_context_detail(layer: str, component_idx: int):
    """Return full activation context data for a single component."""
    try:
        contexts = _ensure_activation_contexts_cached()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    if contexts is None:
        return JSONResponse({"error": "No activation contexts found"}, status_code=404)

    # Raw dict has {"layers": {...}} structure
    layers = contexts.get("layers", contexts)  # Handle both raw dict and direct layers
    layer_data = layers.get(layer)
    if layer_data is None:
        return JSONResponse({"error": f"Layer '{layer}' not found"}, status_code=404)

    for subcomp in layer_data:
        if subcomp["subcomponent_idx"] == component_idx:
            return subcomp

    return JSONResponse(
        {"error": f"Component {component_idx} not found in layer '{layer}'"},
        status_code=404,
    )


@app.get("/api/activation_contexts/subcomponents")
@log_errors
def generate_activation_contexts(
    importance_threshold: float = 0.01,
    n_batches: int = 100,
    batch_size: int = 32,
    n_tokens_either_side: int = 5,
    topk_examples: int = 20,
    separation_tokens: int = 0,
) -> StreamingResponse:
    """Generate activation contexts from training data.

    This streams progress updates and saves the result to DB when complete.
    """
    state = get_state()
    try:
        loaded = get_loaded_run()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    # Create a data loader for generation
    task_config = runtime_cast(LMTaskConfig, loaded.config.task_config)
    train_data_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=loaded.config.tokenizer_name,
        split=task_config.train_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=task_config.shuffle_each_epoch,
        seed=None,
    )
    train_loader, _ = create_data_loader(
        dataset_config=train_data_config,
        batch_size=1,
        buffer_size=task_config.buffer_size,
        global_seed=loaded.config.seed,
    )

    # Create a temporary run context for generation
    # from spd.app.backend.services.run_context_service import TrainRunContext

    # run_context = TrainRunContext(
    # )

    config = ActivationContextsGenerationConfig(
        importance_threshold=importance_threshold,
        n_batches=n_batches,
        batch_size=batch_size,
        n_tokens_either_side=n_tokens_either_side,
        topk_examples=topk_examples,
        separation_tokens=separation_tokens,
    )

    def generate() -> Generator[str]:
        for res in get_activations_data_streaming(
            config=loaded.config,
            cm=loaded.model,
            tokenizer=loaded.tokenizer,
            train_loader=train_loader,
            token_strings=loaded.token_strings,
            importance_threshold=importance_threshold,
            n_batches=n_batches,
            n_tokens_either_side=n_tokens_either_side,
            batch_size=batch_size,
            topk_examples=topk_examples,
            separation_tokens=separation_tokens,
        ):
            match res:
                case ("progress", progress):
                    progress_data = {"type": "progress", "progress": progress}
                    yield f"data: {json.dumps(progress_data)}\n\n"
                case ("complete", data):
                    # Save to DB
                    state.db.set_activation_contexts(loaded.run.id, data, config)

                    # Clear cache so it reloads from DB
                    loaded.activation_contexts_cache = None

                    # Also save to harvest cache for backward compatibility
                    harvest_id = str(uuid.uuid4())
                    harvest_cache[harvest_id] = data

                    metadata = HarvestMetadata(
                        harvest_id=harvest_id,
                        layers={
                            layer_name: [
                                SubcomponentMetadata(
                                    subcomponent_idx=subcomp.subcomponent_idx,
                                    mean_ci=subcomp.mean_ci,
                                )
                                for subcomp in subcomponents
                            ]
                            for layer_name, subcomponents in data.layers.items()
                        },
                    )
                    complete_data = {"type": "complete", "result": metadata.model_dump()}
                    yield f"data: {json.dumps(complete_data)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# -----------------------------------------------------------------------------
# Prompt endpoints
# -----------------------------------------------------------------------------


@app.get("/api/prompts")
@log_errors
def get_prompts():
    """Return list of all prompts for the loaded run."""
    state = get_state()
    assert state.run_state is not None, "No run loaded"

    try:
        loaded = get_loaded_run()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    prompt_ids = state.db.get_all_prompt_ids(loaded.run.id)

    results: list[dict[str, Any]] = []
    for pid in prompt_ids:
        prompt = state.db.get_prompt(pid)
        assert prompt is not None, f"Prompt {pid} in index but not in DB"
        token_strings = [state.run_state.token_strings[t] for t in prompt.token_ids]
        results.append(
            {
                "id": prompt.id,
                "tokens": token_strings,
                "preview": "".join(token_strings[:10]) + ("..." if len(token_strings) > 10 else ""),
            }
        )
    return results


@app.post("/api/prompts/generate")
@log_errors
def generate_prompts(
    n_prompts: int,
) -> StreamingResponse:
    """Generate prompts from training data with CI harvesting.

    Streams progress updates and stores prompts with their active components
    (for the inverted index used by search).

    Args:
        n_prompts: Number of prompts to generate
        seq_length: Sequence length (None = use model's max_seq_len)
        batch_size: Batch size for CI computation (default 32)
        ci_threshold: Threshold for component activation
        output_prob_threshold: Threshold for output token activation
    """

    state = get_state()
    try:
        loaded = get_loaded_run()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    # Create a data loader for generation
    task_config = runtime_cast(LMTaskConfig, loaded.config.task_config)
    actual_seq_length = 8  # seq_length or task_config.max_seq_len

    train_data_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=loaded.config.tokenizer_name,
        split=task_config.train_data_split,
        n_ctx=actual_seq_length,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=task_config.shuffle_each_epoch,
    )
    train_loader, _ = create_data_loader(
        dataset_config=train_data_config,
        batch_size=32,  # somewhat arbitrary, could be made configurable
        buffer_size=task_config.buffer_size,
        global_seed=loaded.config.seed,
    )

    def generate() -> Generator[str]:
        added_count = 0
        from spd.utils.general_utils import extract_batch_data

        for batch in train_loader:
            if added_count >= n_prompts:
                break

            tokens = extract_batch_data(batch).to(DEVICE)
            actual_batch_size = tokens.shape[0]
            n_seq = tokens.shape[1]

            # Compute CI for the whole batch
            ci_result = compute_ci_only(
                model=loaded.model,
                tokens=tokens,
                sampling=loaded.sampling,
            )

            # Process each prompt in the batch
            for i in range(actual_batch_size):
                if added_count >= n_prompts:
                    break

                token_ids = tokens[i].tolist()

                # Slice CI for this single prompt
                ci_single = {k: v[i : i + 1] for k, v in ci_result.ci_lower_leaky.items()}
                probs_single = ci_result.output_probs[i : i + 1]

                # Extract active components for inverted index
                active_components = extract_active_from_ci(
                    ci_lower_leaky=ci_single,
                    output_probs=probs_single,
                    ci_threshold=0.0,  # consider removing this arg entirely
                    output_prob_threshold=0.01,  # TODO change me to topP (cumulative probability threshold)
                    n_seq=n_seq,
                )

                # Add to DB with active components
                state.db.add_prompt(loaded.run.id, token_ids, active_components)
                added_count += 1

            # Stream progress after each batch
            progress = min(added_count / n_prompts, 1.0)
            progress_data = {"type": "progress", "progress": progress, "count": added_count}
            yield f"data: {json.dumps(progress_data)}\n\n"

        # Final result
        total = state.db.get_prompt_count(loaded.run.id)
        complete_data = {
            "type": "complete",
            "prompts_added": added_count,
            "total_prompts": total,
        }
        yield f"data: {json.dumps(complete_data)}\n\n"
        logger.info(f"[API] Generated {added_count} prompts for run {loaded.run.id}")

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/prompt/{prompt_id}")
@log_errors
def get_prompt(
    prompt_id: int,
    max_mean_ci: Annotated[float, Query(ge=0, le=1)] = 1.0,
    normalize: Annotated[bool, Query()] = True,
    ci_threshold: Annotated[float, Query(ge=0)] = 1e-6,
    output_prob_threshold: Annotated[float, Query(ge=0, le=1)] = 0.01,
):
    """Return prompt data with on-demand graph computation."""
    import time

    print(f"[API] /api/prompt/{prompt_id} called")

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
            {
                "error": f"Prompt {prompt_id} belongs to run {prompt.run_id}, not loaded run {loaded.run.id}"
            },
            status_code=400,
        )

    token_ids = prompt.token_ids
    token_strings = [loaded.tokenizer.decode([t]) for t in token_ids]

    tokens_tensor = torch.tensor([token_ids], device=DEVICE)
    print(f"[API] /api/prompt/{prompt_id} Running for prompt {token_strings} on device {DEVICE}")
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

    logger.info(
        f"[API] /api/prompt/{prompt_id} computed in {t_compute_end - t_compute_start:.2f}s, "
        f"{len(result.edges)} edges"
    )

    # Build mean CI lookup for filtering
    mean_ci_lookup: dict[str, float] = {}
    if max_mean_ci < 1.0:
        activation_contexts = _ensure_activation_contexts_cached()
        if activation_contexts:
            for layer, subcomps in activation_contexts.items():
                for subcomp in subcomps:
                    mean_ci_lookup[f"{layer}:{subcomp['subcomponent_idx']}"] = subcomp["mean_ci"]

    edges = result.edges

    if max_mean_ci < 1.0 and mean_ci_lookup:
        edges = [
            e
            for e in edges
            if mean_ci_lookup.get(f"{e[0]}:{e[2]}", 0.0) <= max_mean_ci
            and mean_ci_lookup.get(f"{e[1]}:{e[3]}", 0.0) <= max_mean_ci
        ]

    edges.sort(key=lambda x: abs(x[6]), reverse=True)
    edges = edges[:GLOBAL_EDGE_LIMIT]

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
@log_errors
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
            {
                "error": f"Prompt {prompt_id} belongs to run {prompt.run_id}, not loaded run {loaded.run.id}"
            },
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
        show_progress=False,
    )
    t_end = time.time()

    logger.info(
        f"[API] /api/prompt/{prompt_id}/optimized completed in {t_end - t_start:.2f}s, "
        f"{len(result.edges)} edges"
    )

    edges = result.edges
    edges.sort(key=lambda x: abs(x[6]), reverse=True)
    edges = edges[:GLOBAL_EDGE_LIMIT]

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


# -----------------------------------------------------------------------------
# Custom prompt endpoints
# -----------------------------------------------------------------------------


@app.post("/api/tokenize")
@log_errors
def tokenize_text(text: str):
    """Tokenize text and return tokens for preview."""
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
@log_errors
def compute_custom_prompt(
    token_ids: Annotated[list[int], Body(embed=True)],
    normalize: Annotated[bool, Query()] = True,
    ci_threshold: Annotated[float, Query(ge=0)] = 1e-6,
    output_prob_threshold: Annotated[float, Query(ge=0, le=1)] = 0.01,
):
    """Compute attribution graph for custom token IDs (not stored in DB)."""
    import time

    try:
        loaded = get_loaded_run()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    if not token_ids:
        return JSONResponse({"error": "No token IDs provided"}, status_code=400)

    token_strings = [loaded.tokenizer.decode([t]) for t in token_ids]

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
        show_progress=False,
    )
    t_end = time.time()

    logger.info(
        f"[API] /api/prompt/custom computed in {t_end - t_start:.2f}s, {len(result.edges)} edges"
    )

    edges = result.edges
    edges.sort(key=lambda x: abs(x[6]), reverse=True)
    edges = edges[:GLOBAL_EDGE_LIMIT]

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
        "id": None,
        "tokens": token_strings,
        "edges": edges_dicts,
        "outputProbs": output_probs,
    }


# -----------------------------------------------------------------------------
# Search endpoint
# -----------------------------------------------------------------------------


@app.get("/api/search")
@log_errors
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
                "preview": "".join(token_strings[:10]) + ("..." if len(token_strings) > 10 else ""),
            }
        )

    return {
        "query": {"components": component_list, "mode": mode},
        "count": len(results),
        "results": results,
    }


@app.get("/api/health")
@log_errors
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


def cli(port: int = 8000) -> None:
    """Run the server.

    Args:
        port: Port to serve on (default 8000)
    """
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        "spd.app.backend.server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    fire.Fire(cli)
