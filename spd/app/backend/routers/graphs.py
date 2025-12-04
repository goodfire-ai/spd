"""Graph listing and generation endpoints."""

import json
from collections.abc import Generator
from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse

from spd.app.backend.compute import compute_ci_only, extract_active_from_ci
from spd.app.backend.dependencies import DepLoadedRun, DepStateManager
from spd.app.backend.schemas import GraphPreview, GraphSearchQuery, GraphSearchResponse
from spd.app.backend.utils import log_errors
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.log import logger
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import runtime_cast

router = APIRouter(prefix="/api/graphs", tags=["graphs"])

DEVICE = get_device()


@router.get("")
@log_errors
def list_graphs(manager: DepStateManager, loaded: DepLoadedRun) -> list[GraphPreview]:
    """Return list of all graphs for the loaded run."""
    db = manager.db
    graph_ids = db.get_all_graph_ids(loaded.run.id)

    results: list[GraphPreview] = []
    for gid in graph_ids:
        graph = db.get_graph(gid)
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
    return results


@router.post("/generate")
@log_errors
def generate_graphs(
    n_graphs: int,
    manager: DepStateManager,
    loaded: DepLoadedRun,
) -> StreamingResponse:
    """Generate attribution graphs from training data with CI harvesting.

    Streams progress updates and stores graphs with their active components
    (for the inverted index used by search).
    """
    db = manager.db

    # Create a data loader for generation
    task_config = runtime_cast(LMTaskConfig, loaded.config.task_config)
    actual_seq_length = 8

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
        batch_size=32,
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
                    ci_threshold=0.0,
                    output_prob_threshold=0.01,
                    n_seq=n_seq,
                )

                # Add to DB with active components
                db.add_graph(loaded.run.id, token_ids, active_components)
                added_count += 1

            # Stream progress after each batch
            progress = min(added_count / n_graphs, 1.0)
            progress_data = {"type": "progress", "progress": progress, "count": added_count}
            yield f"data: {json.dumps(progress_data)}\n\n"

        # Final result
        total = db.get_graph_count(loaded.run.id)
        complete_data = {
            "type": "complete",
            "graphs_added": added_count,
            "total_graphs": total,
        }
        yield f"data: {json.dumps(complete_data)}\n\n"
        logger.info(f"[API] Generated {added_count} graphs for run {loaded.run.id}")

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/search")
@log_errors
def search_graphs(
    manager: DepStateManager,
    loaded: DepLoadedRun,
    components: str = "",
    mode: Annotated[str, Query(pattern="^(all|any)$")] = "all",
) -> GraphSearchResponse:
    """Search for attribution graphs with specified components in the loaded run."""
    db = manager.db

    component_list = [c.strip() for c in components.split(",") if c.strip()]
    if not component_list:
        return JSONResponse({"error": "No components specified"}, status_code=400)  # pyright: ignore[reportReturnType]

    require_all = mode == "all"
    graph_ids = db.find_graphs_with_components(loaded.run.id, component_list, require_all=require_all)

    results: list[GraphPreview] = []
    for gid in graph_ids:
        graph = db.get_graph(gid)
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
