"""FastAPI server for querying the local attribution database.

Loads the SPD model at startup and computes attribution graphs on-demand.
This requires GPU for efficient computation.

Usage:
    python -m spd.attributions.server --db_path ./local_attr.db --port 8765

API Endpoints:
    GET /api/meta                  - Database metadata
    GET /api/activation_contexts   - Activation contexts for all components
    GET /api/prompts               - List all prompts (id, tokens preview)
    GET /api/prompt/{id}           - Prompt data with on-demand graph computation
    GET /api/search                - Find prompts with specific components
"""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import torch
from fastapi import FastAPI, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from spd.attributions.compute import compute_local_attributions, get_sources_by_target
from spd.attributions.db import LocalAttrDB
from spd.attributions.edge_normalization import normalize_edges_by_target
from spd.models.component_model import ComponentModel, SPDRunInfo

if TYPE_CHECKING:
    from spd.configs import SamplingType

THIS_DIR = Path(__file__).parent

app = FastAPI(title="Local Attributions API")
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Global references (set at startup)
db: LocalAttrDB | None = None
model: ComponentModel | None = None
tokenizer: PreTrainedTokenizerFast | None = None
sources_by_target: dict[str, list[str]] | None = None
sampling: "SamplingType | None" = None
device: str = "cuda" if torch.cuda.is_available() else "cpu"

# Cache for activation contexts (large, doesn't change)
_activation_contexts_cache: dict[str, Any] | None = None


@app.on_event("startup")
def startup_event():
    """Initialize DB, model, and tokenizer when worker starts."""
    import os
    import sys

    global db, model, tokenizer, sources_by_target, sampling

    db_path_str = os.environ.get("LOCAL_ATTR_DB_PATH")
    assert db_path_str is not None, "LOCAL_ATTR_DB_PATH env var must be set"

    db = LocalAttrDB(Path(db_path_str), check_same_thread=False)
    print(f"[STARTUP] Worker initialized with DB: {db_path_str}", flush=True)
    print(f"[STARTUP] Device: {device}", flush=True)
    print(f"[STARTUP] CUDA available: {torch.cuda.is_available()}", flush=True)
    sys.stdout.flush()

    # Load model from wandb_path stored in DB meta
    wandb_info = db.get_meta("wandb_path")
    n_blocks_info = db.get_meta("n_blocks")
    assert wandb_info is not None, "DB must have wandb_path in meta"
    wandb_path = wandb_info["path"]
    assert wandb_path is not None, "wandb_path must be set"

    assert n_blocks_info is not None, "DB must have n_blocks in meta"
    n_blocks = n_blocks_info["n_blocks"]
    assert isinstance(n_blocks, int), "n_blocks must be int"

    print(f"Loading model from {wandb_path}...")
    run_info = SPDRunInfo.from_path(wandb_path)
    model = ComponentModel.from_run_info(run_info)
    model = model.to(device)
    model.eval()

    # Load tokenizer
    spd_config = run_info.config
    assert spd_config.tokenizer_name is not None
    loaded_tokenizer = AutoTokenizer.from_pretrained(spd_config.tokenizer_name)
    assert isinstance(loaded_tokenizer, PreTrainedTokenizerFast)
    tokenizer = loaded_tokenizer

    # Build sources_by_target mapping
    sampling = spd_config.sampling
    sources_by_target = get_sources_by_target(model, device, sampling, n_blocks)
    print(f"Model loaded on {device}, ready for on-demand computation")


# -----------------------------------------------------------------------------
# API Routes
# -----------------------------------------------------------------------------


@app.get("/api/meta")
def get_meta() -> dict[str, Any]:
    """Return database metadata."""
    print("[API] /api/meta called")
    assert db is not None
    wandb_info = db.get_meta("wandb_path")
    n_blocks_info = db.get_meta("n_blocks")
    result = {
        "wandb_path": wandb_info.get("path") if isinstance(wandb_info, dict) else None,
        "n_blocks": n_blocks_info.get("n_blocks") if isinstance(n_blocks_info, dict) else None,
        "prompt_count": db.get_prompt_count(),
    }
    print(f"[API] /api/meta done: {result}")
    return result


@app.get("/api/activation_contexts")
def get_activation_contexts():
    """Return activation contexts (component metadata). Cached after first load."""
    print("[API] /api/activation_contexts called")
    global _activation_contexts_cache
    assert db is not None

    if _activation_contexts_cache is not None:
        print("[API] /api/activation_contexts returning cached")
        return _activation_contexts_cache

    print("[API] /api/activation_contexts fetching from DB...")
    contexts = db.get_activation_contexts()
    if contexts is None:
        print("[API] /api/activation_contexts not found")
        return JSONResponse({"error": "No activation contexts found"}, status_code=404)

    print(f"[API] /api/activation_contexts done: {len(contexts)} layers")
    _activation_contexts_cache = contexts
    return contexts


@app.get("/api/prompts")
def get_prompts() -> list[dict[str, Any]]:
    """Return list of all prompts (summaries with decoded tokens)."""
    print("[API] /api/prompts called")
    assert db is not None
    assert tokenizer is not None

    prompt_ids = db.get_all_prompt_ids()
    print(f"[API] /api/prompts found {len(prompt_ids)} prompts")
    results: list[dict[str, Any]] = []
    for pid in prompt_ids:
        prompt = db.get_prompt_simple(pid)
        assert prompt is not None, f"Prompt {pid} in index but not in DB"
        token_strings = [tokenizer.decode([t]) for t in prompt.token_ids]
        results.append(
            {
                "id": prompt.id,
                "tokens": token_strings,
                "preview": "".join(token_strings[:10]) + ("..." if len(token_strings) > 10 else ""),
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
    """Return prompt data with on-demand graph computation.

    Computes the full attribution graph for this prompt on-demand,
    then applies server-side filtering.

    Args:
        prompt_id: The prompt ID to fetch
        max_mean_ci: Filter out edges where either endpoint has mean_ci > this value
        normalize: If True, normalize incoming edges to each node to sum to 1
        ci_threshold: Threshold for considering a component alive
        output_prob_threshold: Threshold for considering an output token alive
    """
    import time

    print(
        f"[API] /api/prompt/{prompt_id} called"
        f"\n  - ci_threshold={ci_threshold}"
        f"\n  - output_prob_threshold={output_prob_threshold}"
        f"\n  - normalizing={normalize}"
        f"\n  - max_mean_ci={max_mean_ci}",
        flush=True,
    )
    assert db is not None
    assert model is not None
    assert tokenizer is not None
    assert sources_by_target is not None
    assert sampling is not None

    # Get token IDs from DB
    prompt = db.get_prompt_simple(prompt_id)
    if prompt is None:
        return JSONResponse({"error": f"Prompt {prompt_id} not found"}, status_code=404)
    
    CLAMP = 6
    token_ids = prompt.token_ids[:CLAMP]

    # Decode tokens for display
    token_strings = [tokenizer.decode([t]) for t in token_ids]
    print(
        f"[API] /api/prompt/{prompt_id} computing attributions for {len(token_strings)} tokens on device={device}...",
        flush=True,
    )

    # Compute attribution graph on-demand (needs gradients for autograd.grad)
    tokens_tensor = torch.tensor([token_ids], device=device)
    t_compute_start = time.time()
    result = compute_local_attributions(
        model=model,
        tokens=tokens_tensor,
        sources_by_target=sources_by_target,
        ci_threshold=ci_threshold,
        output_prob_threshold=output_prob_threshold,
        sampling=sampling,
        device=device,
        show_progress=True,
    )
    t_compute_end = time.time()

    print(
        f"[API] /api/prompt/{prompt_id} graph computed in {t_compute_end - t_compute_start:.2f}s, {len(result.edges)} edges",
        flush=True,
    )

    # # Get mean CI lookup for filtering (if available)
    # mean_ci_lookup: dict[str, float] = {}
    if max_mean_ci < 1.0:
        print("WARN max mean ci filtering is not implemented atm")
        # activation_contexts = db.get_activation_contexts()
        # if activation_contexts:
        #     for layer, subcomps in activation_contexts.items():
        #         for subcomp in subcomps:
        #             mean_ci_lookup[f"{layer}:{subcomp['subcomponent_idx']}"] = subcomp["mean_ci"]

    edges = result.edges

    edges.sort(key=lambda x: abs(x[6]), reverse=True)
    edges = edges[:30_000]

    print(f"[API] /api/prompt/{prompt_id} got {len(edges)} edges")

    # Normalize edges before filtering (if enabled)
    if normalize:
        print("Normalizing edges by target")
        edges = normalize_edges_by_target(edges)
        print("  done")

    # Format edges for client
    print("Formatting edges for client")
    edges_dicts = [
        {
            "src": f"{e[0]}:{e[4]}:{e[2]}",  # source:s_in:c_in_idx
            "tgt": f"{e[1]}:{e[5]}:{e[3]}",  # target:s_out:c_out_idx
            "val": e[6],
        }
        for e in edges
    ]
    print("  done")

    # Extract output_probs for display (tokens above threshold at each position)
    output_probs: dict[str, dict[str, float | str]] = {}
    output_probs_tensor = result.output_probs[0].cpu()  # [seq, vocab]

    print("Extracting output probs for display")
    for s in range(output_probs_tensor.shape[0]):
        for c_idx in range(output_probs_tensor.shape[1]):
            prob = float(output_probs_tensor[s, c_idx].item())
            # if prob >= output_prob_threshold:
            key = f"{s}:{c_idx}"
            output_probs[key] = {"prob": round(prob, 6), "token": tokenizer.decode([c_idx])}
    print("  done")

    return {
        "id": prompt.id,
        "tokens": token_strings,
        "edges": edges_dicts,
        "outputProbs": output_probs,
    }


@app.get("/api/search")
def search_prompts(
    components: str = "",
    mode: Annotated[str, Query(pattern="^(all|any)$")] = "all",
):
    """Search for prompts with specified components.

    Args:
        components: Comma-separated component keys like "h.0.attn.q_proj:5,h.1.mlp.c_fc:10"
        mode: "all" requires all components, "any" requires at least one
    """
    assert db is not None
    assert tokenizer is not None

    component_list = [c.strip() for c in components.split(",") if c.strip()]
    if not component_list:
        return JSONResponse({"error": "No components specified"}, status_code=400)

    require_all = mode == "all"
    prompt_ids = db.find_prompts_with_components(component_list, require_all=require_all)

    # Get prompt data and decode tokens for matching prompts
    results: list[dict[str, Any]] = []
    for pid in prompt_ids:
        prompt = db.get_prompt_simple(pid)
        assert prompt is not None, f"Prompt {pid} in index but not in DB"
        token_strings = [tokenizer.decode([t]) for t in prompt.token_ids]
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


def create_app(db_path: Path) -> FastAPI:
    """Create the FastAPI app with DB initialized."""
    global db
    # check_same_thread=False is safe here since we're only reading
    db = LocalAttrDB(db_path, check_same_thread=False)

    # Verify DB has data
    count = db.get_prompt_count()
    print(f"Database loaded: {count} prompts")

    return app


def main(db_path: str, port: int = 8765, workers: int = 1):
    """Run the server.

    Args:
        db_path: Path to SQLite database
        port: Port to serve on (default 8765)
        host: Host to bind to (default localhost)
        workers: Number of uvicorn workers. Each loads the model, so 1 is usually best.
    """
    import uvicorn

    db_path_ = Path(db_path)
    assert db_path_.exists(), f"Database not found: {db_path_}"

    # Store db_path for worker initialization
    import os
    import signal

    os.environ["LOCAL_ATTR_DB_PATH"] = str(db_path_)

    def force_exit(sig, frame):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType, reportUnusedParameter]
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
