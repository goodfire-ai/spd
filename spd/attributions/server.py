"""FastAPI server for querying the local attribution database.

Usage:
    python -m spd.attributions.server --db_path ./local_attr.db --port 8765

API Endpoints:
    GET /api/meta                  - Database metadata
    GET /api/activation_contexts   - Activation contexts for all components
    GET /api/prompts               - List all prompts (id, tokens preview)
    GET /api/prompt/{id}           - Prompt data with server-side top-k filtering
    GET /api/search                - Find prompts with specific components
"""

import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse

from spd.attributions.db import LocalAttrDB
from spd.attributions.edge_normalization import normalize_edges_by_target

THIS_DIR = Path(__file__).parent

app = FastAPI(title="Local Attributions API")
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Global DB reference (set in lifespan or startup event)
db: LocalAttrDB | None = None

# Cache for activation contexts (large, doesn't change)
_activation_contexts_cache: dict[str, Any] | None = None


@app.on_event("startup")
def startup_event():
    """Initialize DB connection when worker starts."""
    import os

    global db
    if db is None:
        db_path = os.environ.get("LOCAL_ATTR_DB_PATH")
        if db_path:
            db = LocalAttrDB(Path(db_path), check_same_thread=False)
            print(f"Worker initialized with DB: {db_path}")


# -----------------------------------------------------------------------------
# API Routes
# -----------------------------------------------------------------------------


@app.get("/api/meta")
def get_meta() -> dict[str, Any]:
    """Return database metadata."""
    assert db is not None
    wandb_info = db.get_meta("wandb_path")
    n_blocks_info = db.get_meta("n_blocks")
    return {
        "wandb_path": wandb_info.get("path") if isinstance(wandb_info, dict) else None,
        "n_blocks": n_blocks_info.get("n_blocks") if isinstance(n_blocks_info, dict) else None,
        "prompt_count": db.get_prompt_count(),
    }


@app.get("/api/activation_contexts")
def get_activation_contexts():
    """Return activation contexts (component metadata). Cached after first load."""
    global _activation_contexts_cache
    assert db is not None

    if _activation_contexts_cache is not None:
        return _activation_contexts_cache

    contexts = db.get_activation_contexts()
    if contexts is None:
        return JSONResponse({"error": "No activation contexts found"}, status_code=404)

    _activation_contexts_cache = contexts
    return contexts


@app.get("/api/prompts")
def get_prompts():
    """Return list of all prompts (summaries)."""
    assert db is not None
    summaries = db.get_all_prompt_summaries()
    return [
        {
            "id": s.id,
            "tokens": s.tokens,
            "preview": "".join(s.tokens[:10]) + ("..." if len(s.tokens) > 10 else ""),
        }
        for s in summaries
    ]


@app.get("/api/prompt/{prompt_id}")
def get_prompt(
    prompt_id: int,
    top_k: Annotated[int, Query(ge=10, le=50000)] = 1000,
    max_mean_ci: Annotated[float, Query(ge=0, le=1)] = 1.0,
    normalize: Annotated[bool, Query()] = True,
):
    """Return prompt data with server-side top-k edge filtering.

    Args:
        prompt_id: The prompt ID to fetch
        top_k: Maximum number of edges to return (sorted by |attribution|)
        max_mean_ci: Filter out edges where either endpoint has mean_ci > this value
        normalize: If True, normalize incoming edges to each node to sum to 1
    """
    assert db is not None
    prompt = db.get_prompt(prompt_id)
    if prompt is None:
        return JSONResponse({"error": f"Prompt {prompt_id} not found"}, status_code=404)

    pairs = json.loads(prompt.pairs_json)

    # Get mean CI lookup for filtering (if available)
    mean_ci_lookup: dict[str, float] = {}
    if max_mean_ci < 1.0:
        activation_contexts = db.get_activation_contexts()
        if activation_contexts:
            for layer, subcomps in activation_contexts.items():
                for subcomp in subcomps:
                    mean_ci_lookup[f"{layer}:{subcomp['subcomponent_idx']}"] = subcomp[
                        "mean_ci"
                    ]

    # Extract all edges, filter by CI, sort by |val|, take top_k
    all_edges: list[tuple[str, str, int, int, int, int, float, bool]] = []
    # (source, target, c_in_idx, c_out_idx, s_in, s_out, val, is_cross_seq)

    for pair in pairs:
        source = pair["source"]
        target = pair["target"]
        is_cross_seq = pair["is_cross_seq"]
        c_in_idxs = pair["trimmed_c_in_idxs"]
        c_out_idxs = pair["trimmed_c_out_idxs"]

        for entry in pair["attribution"]:
            if is_cross_seq:
                s_in, c_in_local, s_out, c_out_local, val = entry
            else:
                s_in = s_out = entry[0]
                c_in_local = entry[1]
                c_out_local = entry[2]
                val = entry[3]

            c_in_idx = c_in_idxs[c_in_local]
            c_out_idx = c_out_idxs[c_out_local]

            # Filter by mean CI
            if max_mean_ci < 1.0:
                src_ci = mean_ci_lookup.get(f"{source}:{c_in_idx}", 0)
                tgt_ci = mean_ci_lookup.get(f"{target}:{c_out_idx}", 0)
                if src_ci > max_mean_ci or tgt_ci > max_mean_ci:
                    continue

            all_edges.append(
                (source, target, c_in_idx, c_out_idx, s_in, s_out, val, is_cross_seq)
            )

    # Normalize edges before filtering (if enabled)
    if normalize:
        all_edges = normalize_edges_by_target(all_edges)

    # Sort by |val| descending, take top_k
    all_edges.sort(key=lambda e: abs(e[6]), reverse=True)
    top_edges = all_edges[:top_k]

    # Group back into pairs format for client compatibility
    # But use a simpler flat edge format that's easier to process
    edges = [
        {
            "src": f"{e[0]}:{e[4]}:{e[2]}",  # source:s_in:c_in_idx
            "tgt": f"{e[1]}:{e[5]}:{e[3]}",  # target:s_out:c_out_idx
            "val": e[6],
        }
        for e in top_edges
    ]

    return {
        "id": prompt.id,
        "tokens": prompt.tokens,
        "edges": edges,
        "output_probs": prompt.output_probs,
    }


@app.get("/api/search")
def search_prompts(  # pyright: ignore[reportUnknownParameterType]
    components: str = "",
    mode: Annotated[str, Query(pattern="^(all|any)$")] = "all",
):
    """Search for prompts with specified components.

    Args:
        components: Comma-separated component keys like "h.0.attn.q_proj:5,h.1.mlp.c_fc:10"
        mode: "all" requires all components, "any" requires at least one
    """
    assert db is not None
    component_list = [c.strip() for c in components.split(",") if c.strip()]
    if not component_list:
        return JSONResponse({"error": "No components specified"}, status_code=400)

    require_all = mode == "all"
    prompt_ids = db.find_prompts_with_components(component_list, require_all=require_all)

    # Get summaries for matching prompts
    all_summaries = {s.id: s for s in db.get_all_prompt_summaries()}
    results = []
    for pid in prompt_ids:
        if pid in all_summaries:
            s = all_summaries[pid]
            results.append(
                {
                    "id": s.id,
                    "tokens": s.tokens,
                    "preview": "".join(s.tokens[:10])
                    + ("..." if len(s.tokens) > 10 else ""),
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


def main(db_path: str, port: int = 8765, host: str = "localhost", workers: int = 4):
    """Run the server.

    Args:
        db_path: Path to SQLite database
        port: Port to serve on (default 8765)
        host: Host to bind to (default localhost)
        workers: Number of worker processes (default 4)
    """
    import uvicorn

    db_path_ = Path(db_path)
    assert db_path_.exists(), f"Database not found: {db_path_}"

    # Store db_path for worker initialization
    import os

    os.environ["LOCAL_ATTR_DB_PATH"] = str(db_path_)

    print(f"Server running at http://{host}:{port}/")
    print(f"  Alpine.js UI: http://{host}:{port}/")
    print(f"  Workers: {workers}")

    uvicorn.run(
        "spd.attributions.server:app",
        host=host,
        port=port,
        log_level="warning",
        workers=workers,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
