"""Unified FastAPI server for the SPD app.

Merges the main app backend with the local attributions server.
Supports multiple runs, on-demand attribution graph computation,
and activation contexts generation.

Usage:
    python -m spd.app.backend.server --port 8000
"""

import functools
import json
import queue
import re
import threading
import traceback
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
from spd.app.backend.lib.activation_contexts import get_activations_data
from spd.app.backend.lib.edge_normalization import normalize_edges_by_target
from spd.app.backend.optim_cis.run_optim_cis import OptimCIConfig
from spd.app.backend.schemas import (
    ActivationContextsGenerationConfig,
    EdgeData,
    GraphData,
    GraphDataWithOptimization,
    GraphPreview,
    GraphSearchQuery,
    GraphSearchResponse,
    HarvestMetadata,
    LoadedRun,
    ModelActivationContexts,
    OptimizationResult,
    OutputProbability,
    RunInfo,
    SubcomponentActivationContexts,
    SubcomponentMetadata,
    TokenizeResponse,
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


# Expected format from frontend: entity/project/runId (8-char lowercase alphanumeric)
WANDB_PATH_RE = re.compile(r"^([^/\s]+)/([^/\s]+)/([a-z0-9]{8})$")


def validate_wandb_path(path: str) -> tuple[str, str, str]:
    """Validate that path is in expected entity/project/runId format.

    The frontend handles all format parsing and normalization.
    Backend just validates the expected normalized format.

    Returns (entity, project, run_id) tuple.
    """
    m = WANDB_PATH_RE.match(path.strip())
    if not m:
        raise ValueError(
            f'Invalid W&B path format. Expected "entity/project/runId" '
            f"(8-char lowercase alphanumeric run id). Got: {path}"
        )
    return m.groups()  # pyright: ignore[reportReturnType]


@dataclass
class RunState:
    """Runtime state for a loaded run (model, tokenizer, etc.)"""

    run: Run
    model: ComponentModel
    tokenizer: PreTrainedTokenizerBase
    sources_by_target: dict[str, list[str]]
    config: Config
    token_strings: dict[int, str]
    activation_contexts_cache: ModelActivationContexts | None = None


@dataclass
class AppState:
    """Server state. DB is always available; loaded_run is set after /api/runs/load."""

    db: LocalAttrDB
    run_state: RunState | None = field(default=None)


_state: AppState | None = None


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
            graph_count=state.db.get_graph_count(run.id),
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
        # this should be the only place we use the tokenizer to decode. elsewhere we use the token
        # lookup
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

    Expects path in normalized format: entity/project/runId
    (Frontend handles parsing various formats into this normalized form)

    This loads the model onto GPU and makes it available for attribution computation.
    """
    state = get_state()

    # Validate the path format (frontend has already normalized it)
    try:
        entity, project, run_id = validate_wandb_path(unquote(wandb_path))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # Construct full path as stored in DB: wandb:entity/project/runs/runid
    full_wandb_path = f"wandb:{entity}/{project}/runs/{run_id}"

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
    sources_by_target = get_sources_by_target(model, DEVICE, spd_config.sampling)

    # Build token lookup for activation contexts
    token_strings = _build_token_lookup(loaded_tokenizer, spd_config.tokenizer_name)

    state.run_state = RunState(
        run=run,
        model=model,
        tokenizer=loaded_tokenizer,
        sources_by_target=sources_by_target,
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
        has_graphs=state.db.has_graphs(run.id),
        graph_count=state.db.get_graph_count(run.id),
    )


# -----------------------------------------------------------------------------
# Activation contexts endpoints
# -----------------------------------------------------------------------------


def _ensure_activation_contexts_cached() -> ModelActivationContexts | None:
    """Load activation contexts into cache if not already loaded."""
    loaded = get_loaded_run()
    state = get_state()
    if loaded.activation_contexts_cache is None:
        contexts = state.db.get_activation_contexts(loaded.run.id)
        if contexts is not None:
            loaded.activation_contexts_cache = contexts
    return loaded.activation_contexts_cache


@app.get("/api/activation_contexts/summary")
@log_errors
def get_activation_contexts_summary() -> dict[str, list[SubcomponentMetadata]]:
    """Return lightweight summary of activation contexts (just idx + mean_ci per component)."""
    try:
        contexts = _ensure_activation_contexts_cached()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)  # pyright: ignore[reportReturnType]

    if contexts is None:
        return JSONResponse(  # pyright: ignore[reportReturnType]
            {"error": "No activation contexts found. Generate them first.", "missing": True},
            status_code=404,
        )

    summary: dict[str, list[SubcomponentMetadata]] = {}
    for layer, subcomps in contexts.layers.items():
        summary[layer] = [
            SubcomponentMetadata(subcomponent_idx=s.subcomponent_idx, mean_ci=s.mean_ci)
            for s in subcomps
        ]
    return summary


@app.get("/api/activation_contexts/{layer}/{component_idx}")
@log_errors
def get_activation_context_detail(layer: str, component_idx: int) -> SubcomponentActivationContexts:
    """Return full activation context data for a single component."""
    try:
        contexts = _ensure_activation_contexts_cached()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)  # pyright: ignore[reportReturnType]

    if contexts is None:
        return JSONResponse({"error": "No activation contexts found"}, status_code=404)  # pyright: ignore[reportReturnType]

    layer_data = contexts.layers.get(layer)
    if layer_data is None:
        return JSONResponse({"error": f"Layer '{layer}' not found"}, status_code=404)  # pyright: ignore[reportReturnType]

    for subcomp in layer_data:
        if subcomp.subcomponent_idx == component_idx:
            return subcomp

    return JSONResponse(  # pyright: ignore[reportReturnType]
        {"error": f"Component {component_idx} not found in layer '{layer}'"},
        status_code=404,
    )


@app.get("/api/activation_contexts/subcomponents")
@log_errors
def generate_activation_contexts(
    importance_threshold: Annotated[float, Query(gt=0, le=1)],
    n_batches: Annotated[int, Query(gt=0)],
    batch_size: Annotated[int, Query(gt=0)],
    n_tokens_either_side: Annotated[int, Query(ge=0)],
    topk_examples: Annotated[int, Query(gt=0)],
    separation_tokens: Annotated[int, Query(ge=0)],
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

    config = ActivationContextsGenerationConfig(
        importance_threshold=importance_threshold,
        n_batches=n_batches,
        batch_size=batch_size,
        n_tokens_either_side=n_tokens_either_side,
        topk_examples=topk_examples,
        separation_tokens=separation_tokens,
    )

    progress_queue: queue.Queue[dict[str, Any]] = queue.Queue()

    def on_progress(progress: float) -> None:
        progress_queue.put({"type": "progress", "progress": progress})

    def compute_thread() -> None:
        try:
            act_contexts = get_activations_data(
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
                onprogress=on_progress,
            )
            state.db.set_activation_contexts(loaded.run.id, act_contexts, config)

            # Clear cache so it reloads from DB
            loaded.activation_contexts_cache = None

            metadata = HarvestMetadata(
                layers={
                    layer_name: [
                        SubcomponentMetadata(
                            subcomponent_idx=subcomp.subcomponent_idx,
                            mean_ci=subcomp.mean_ci,
                        )
                        for subcomp in subcomponents
                    ]
                    for layer_name, subcomponents in act_contexts.layers.items()
                },
            )
            progress_queue.put({"type": "complete", "result": metadata.model_dump()})
        except Exception as e:
            progress_queue.put({"type": "error", "error": str(e)})

    def generate() -> Generator[str]:
        thread = threading.Thread(target=compute_thread)
        thread.start()

        while True:
            try:
                msg = progress_queue.get(timeout=0.1)
            except queue.Empty:
                if not thread.is_alive():
                    break
                continue

            if msg["type"] == "progress":
                yield f"data: {json.dumps(msg)}\n\n"
            elif msg["type"] == "error" or msg["type"] == "complete":
                yield f"data: {json.dumps(msg)}\n\n"
                break

        thread.join()

    return StreamingResponse(generate(), media_type="text/event-stream")


# -----------------------------------------------------------------------------
# Graph listing endpoints
# -----------------------------------------------------------------------------


@app.get("/api/graphs")
@log_errors
def list_graphs() -> list[GraphPreview]:
    """Return list of all graphs for the loaded run."""
    state = get_state()
    assert state.run_state is not None, "No run loaded"

    try:
        loaded = get_loaded_run()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)  # pyright: ignore[reportReturnType]

    graph_ids = state.db.get_all_graph_ids(loaded.run.id)

    results: list[GraphPreview] = []
    for gid in graph_ids:
        graph = state.db.get_graph(gid)
        assert graph is not None, f"Graph {gid} in index but not in DB"
        token_strings = [state.run_state.token_strings[t] for t in graph.token_ids]
        results.append(
            GraphPreview(
                id=graph.id,
                token_ids=graph.token_ids,
                tokens=token_strings,
                preview="".join(token_strings[:10]) + ("..." if len(token_strings) > 10 else ""),
            )
        )
    return results


@app.post("/api/graphs/generate")
@log_errors
def generate_graphs(
    n_graphs: int,
) -> StreamingResponse:
    """Generate attribution graphs from training data with CI harvesting.

    Streams progress updates and stores graphs with their active components
    (for the inverted index used by search).

    Args:
        n_graphs: Number of graphs to generate
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
            if added_count >= n_graphs:
                break

            tokens = extract_batch_data(batch).to(DEVICE)
            actual_batch_size = tokens.shape[0]
            n_seq = tokens.shape[1]

            # Compute CI for the whole batch
            ci_result = compute_ci_only(
                model=loaded.model,
                tokens=tokens,
                sampling=loaded.config.sampling,
            )

            # Process each sequence in the batch
            for i in range(actual_batch_size):
                if added_count >= n_graphs:
                    break

                token_ids = tokens[i].tolist()

                # Slice CI for this single sequence
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
                state.db.add_graph(loaded.run.id, token_ids, active_components)
                added_count += 1

            # Stream progress after each batch
            progress = min(added_count / n_graphs, 1.0)
            progress_data = {"type": "progress", "progress": progress, "count": added_count}
            yield f"data: {json.dumps(progress_data)}\n\n"

        # Final result
        total = state.db.get_graph_count(loaded.run.id)
        complete_data = {
            "type": "complete",
            "graphs_added": added_count,
            "total_graphs": total,
        }
        yield f"data: {json.dumps(complete_data)}\n\n"
        logger.info(f"[API] Generated {added_count} graphs for run {loaded.run.id}")

    return StreamingResponse(generate(), media_type="text/event-stream")


# -----------------------------------------------------------------------------
# Tokenize and compute endpoints
# -----------------------------------------------------------------------------


@app.post("/api/tokenize")
@log_errors
def tokenize_text(text: str) -> TokenizeResponse:
    """Tokenize text and return tokens for preview (special tokens filtered)."""
    try:
        loaded = get_loaded_run()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)  # pyright: ignore[reportReturnType]

    token_ids = loaded.tokenizer.encode(text, add_special_tokens=False)

    return TokenizeResponse(
        text=text,
        token_ids=token_ids,
        tokens=[loaded.token_strings[t] for t in token_ids],
    )


@app.post("/api/compute")
@log_errors
def compute_graph(
    token_ids: Annotated[list[int], Body(embed=True)],
    normalize: Annotated[bool, Query()],
):
    """Compute attribution graph for given token IDs."""
    import time

    ci_threshold = 1e-6
    output_prob_threshold = 0.01

    try:
        loaded = get_loaded_run()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    if not token_ids:
        return JSONResponse({"error": "No token IDs provided"}, status_code=400)

    token_strings = [loaded.token_strings[t] for t in token_ids]

    tokens_tensor = torch.tensor([token_ids], device=DEVICE)
    t_start = time.time()
    result = compute_local_attributions(
        model=loaded.model,
        tokens=tokens_tensor,
        sources_by_target=loaded.sources_by_target,
        ci_threshold=ci_threshold,
        output_prob_threshold=output_prob_threshold,
        sampling=loaded.config.sampling,
        device=DEVICE,
        show_progress=False,
    )
    t_end = time.time()

    logger.info(
        f"[API] /api/compute completed in {t_end - t_start:.2f}s, {len(result.edges)} edges"
    )

    edges = result.edges
    edges.sort(key=lambda x: abs(x[6]), reverse=True)
    edges = edges[:GLOBAL_EDGE_LIMIT]

    if normalize:
        edges = normalize_edges_by_target(edges)

    edges_typed = [
        EdgeData(src=f"{e[0]}:{e[4]}:{e[2]}", tgt=f"{e[1]}:{e[5]}:{e[3]}", val=e[6]) for e in edges
    ]

    output_probs: dict[str, OutputProbability] = {}
    output_probs_tensor = result.output_probs[0].cpu()

    # Only send output probs above threshold to avoid sending entire vocab
    for s in range(output_probs_tensor.shape[0]):
        for c_idx in range(output_probs_tensor.shape[1]):
            prob = float(output_probs_tensor[s, c_idx].item())
            if prob < output_prob_threshold:
                continue
            key = f"{s}:{c_idx}"
            output_probs[key] = OutputProbability(
                prob=round(prob, 6),
                token=loaded.token_strings[c_idx],
            )

    return GraphData(
        id=-1,  # Custom prompts have no ID
        tokens=token_strings,
        edges=edges_typed,
        outputProbs=output_probs,
    )


@app.post("/api/compute/optimized/stream")
@log_errors
def compute_graph_optimized_stream(
    token_ids: Annotated[list[int], Body(embed=True)],
    label_token: Annotated[int, Query()],
    imp_min_coeff: Annotated[float, Query(gt=0)],
    ce_loss_coeff: Annotated[float, Query(gt=0)],
    steps: Annotated[int, Query(gt=0)],
    pnorm: Annotated[float, Query(gt=0, le=1)],
    normalize: Annotated[bool, Query()],
    output_prob_threshold: Annotated[float, Query(ge=0, le=1)],
):
    """Compute optimized attribution graph for given token IDs with streaming progress."""

    lr = 1e-2
    ci_threshold = 1e-6

    try:
        loaded = get_loaded_run()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    if not token_ids:
        return JSONResponse({"error": "No token IDs provided"}, status_code=400)

    tokens_tensor = torch.tensor([token_ids], device=DEVICE)
    label_str = loaded.token_strings[label_token]
    token_strings = [loaded.token_strings[t] for t in token_ids]

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
        sampling=loaded.config.sampling,
        ce_kl_rounding_threshold=0.5,
    )

    progress_queue: queue.Queue[dict[str, Any]] = queue.Queue()

    def on_progress(current: int, total: int, stage: str) -> None:
        progress_queue.put({"type": "progress", "current": current, "total": total, "stage": stage})

    def compute_thread() -> None:
        try:
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
                on_progress=on_progress,
            )
            progress_queue.put({"type": "result", "result": result})
        except Exception as e:
            progress_queue.put({"type": "error", "error": str(e)})

    def generate() -> Generator[str]:
        thread = threading.Thread(target=compute_thread)
        thread.start()

        while True:
            try:
                msg = progress_queue.get(timeout=0.1)
            except queue.Empty:
                if not thread.is_alive():
                    break
                continue

            if msg["type"] == "progress":
                yield f"data: {json.dumps(msg)}\n\n"
            elif msg["type"] == "error":
                yield f"data: {json.dumps(msg)}\n\n"
                break
            elif msg["type"] == "result":
                result = msg["result"]

                edges = result.edges
                edges.sort(key=lambda x: abs(x[6]), reverse=True)
                edges = edges[:GLOBAL_EDGE_LIMIT]

                if normalize:
                    edges = normalize_edges_by_target(edges)

                edges_typed = [
                    EdgeData(src=f"{e[0]}:{e[4]}:{e[2]}", tgt=f"{e[1]}:{e[5]}:{e[3]}", val=e[6])
                    for e in edges
                ]

                output_probs: dict[str, OutputProbability] = {}
                output_probs_tensor = result.output_probs[0].cpu()

                for s in range(output_probs_tensor.shape[0]):
                    for c_idx in range(output_probs_tensor.shape[1]):
                        prob = float(output_probs_tensor[s, c_idx].item())
                        if prob < output_prob_threshold:
                            continue
                        key = f"{s}:{c_idx}"
                        output_probs[key] = OutputProbability(
                            prob=round(prob, 6),
                            token=loaded.token_strings[c_idx],
                        )

                response_data = GraphDataWithOptimization(
                    id=-1,  # Custom prompts have no ID
                    tokens=token_strings,
                    edges=edges_typed,
                    outputProbs=output_probs,
                    optimization=OptimizationResult(
                        label_token=label_token,
                        label_str=label_str,
                        imp_min_coeff=imp_min_coeff,
                        ce_loss_coeff=ce_loss_coeff,
                        steps=steps,
                        label_prob=result.stats.label_prob,
                        l0_total=result.stats.l0_total,
                        l0_per_layer=result.stats.l0_per_layer,
                    ),
                )
                complete_data = {"type": "complete", "data": response_data.model_dump()}
                yield f"data: {json.dumps(complete_data)}\n\n"
                break

        thread.join()

    return StreamingResponse(generate(), media_type="text/event-stream")


# -----------------------------------------------------------------------------
# Search endpoint
# -----------------------------------------------------------------------------


@app.get("/api/search")
@log_errors
def search_graphs(
    components: str = "",
    mode: Annotated[str, Query(pattern="^(all|any)$")] = "all",
) -> GraphSearchResponse:
    """Search for attribution graphs with specified components in the loaded run."""
    state = get_state()
    try:
        loaded = get_loaded_run()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)  # pyright: ignore[reportReturnType]

    component_list = [c.strip() for c in components.split(",") if c.strip()]
    if not component_list:
        return JSONResponse({"error": "No components specified"}, status_code=400)  # pyright: ignore[reportReturnType]

    require_all = mode == "all"
    graph_ids = state.db.find_graphs_with_components(
        loaded.run.id, component_list, require_all=require_all
    )

    results: list[GraphPreview] = []
    for gid in graph_ids:
        graph = state.db.get_graph(gid)
        assert graph is not None, f"Graph {gid} in index but not in DB"
        token_strings = [loaded.token_strings[t] for t in graph.token_ids]
        results.append(
            GraphPreview(
                id=graph.id,
                token_ids=graph.token_ids,
                tokens=token_strings,
                preview="".join(token_strings[:10]) + ("..." if len(token_strings) > 10 else ""),
            )
        )

    return GraphSearchResponse(
        query=GraphSearchQuery(components=component_list, mode=mode),
        count=len(results),
        results=results,
    )


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
