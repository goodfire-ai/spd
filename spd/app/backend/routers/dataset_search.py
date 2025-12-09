"""Dataset search endpoints for SimpleStories exploration.

This module provides search functionality for the SimpleStories dataset,
independent of any loaded SPD run. Results are cached in memory for pagination.
"""

import json
import queue
import threading
import time
from collections.abc import Generator
from typing import Annotated, Any

from datasets import Dataset, load_dataset
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from spd.app.backend.dependencies import DepStateManager
from spd.app.backend.schemas import (
    DatasetSearchMetadata,
    DatasetSearchPage,
    DatasetSearchResult,
)
from spd.app.backend.state import DatasetSearchState
from spd.app.backend.utils import log_errors
from spd.log import logger

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


@router.post("/search")
@log_errors
def search_dataset(
    query: Annotated[str, Query(min_length=1)],
    split: Annotated[str, Query(pattern="^(train|test)$")] = "train",
    manager: DepStateManager = None,  # pyright: ignore[reportArgumentType]
) -> StreamingResponse:
    """Search SimpleStories dataset for stories containing query string.

    Streams progress updates during filtering and caches results for pagination.
    Works independently of any loaded run.

    Args:
        query: Text to search for (case-insensitive)
        split: Dataset split to search ("train" or "test")

    Returns:
        SSE stream with progress updates and final metadata
    """
    progress_queue: queue.Queue[dict[str, Any]] = queue.Queue()

    def search_thread() -> None:
        try:
            start_time = time.time()
            search_query = query.lower()

            # Load dataset
            progress_queue.put({"type": "progress", "progress": 0.1})
            logger.info(f"Loading SimpleStories dataset (split={split})...")
            dataset = load_dataset("lennart-finke/SimpleStories", split=split)
            assert isinstance(dataset, Dataset), f"Expected Dataset, got {type(dataset)}"

            total_stories = len(dataset)
            logger.info(f"Searching {total_stories} stories for '{query}'...")

            progress_queue.put({"type": "progress", "progress": 0.2})

            # Filter using optimized HuggingFace filter with multiprocessing
            filtered = dataset.filter(
                lambda x: search_query in x["story"].lower(),
                num_proc=8,
            )

            progress_queue.put({"type": "progress", "progress": 0.9})

            # Build results list with occurrence counts
            results: list[dict[str, Any]] = []
            for item in filtered:
                item_dict: dict[str, Any] = dict(item)
                story: str = item_dict["story"]
                results.append(
                    {
                        "story": story,
                        "occurrence_count": story.lower().count(search_query),
                        "topic": item_dict.get("topic"),
                        "theme": item_dict.get("theme"),
                    }
                )

            progress_queue.put({"type": "progress", "progress": 0.95})

            search_time = time.time() - start_time

            # Cache results in state manager
            metadata = {
                "query": query,
                "split": split,
                "total_results": len(results),
                "search_time_seconds": search_time,
            }
            manager.state.dataset_search_state = DatasetSearchState(
                results=results,
                metadata=metadata,
            )

            logger.info(
                f"Found {len(results)} results in {search_time:.2f}s "
                f"(searched {total_stories} stories)"
            )

            progress_queue.put(
                {
                    "type": "complete",
                    "metadata": metadata,
                }
            )
        except Exception as e:
            logger.error(f"Dataset search failed: {e}")
            progress_queue.put({"type": "error", "error": str(e)})

    def generate() -> Generator[str]:
        thread = threading.Thread(target=search_thread)
        thread.start()

        while True:
            try:
                msg = progress_queue.get(timeout=0.1)
            except queue.Empty:
                if not thread.is_alive():
                    # Thread exited - drain any remaining messages
                    while not progress_queue.empty():
                        try:
                            msg = progress_queue.get_nowait()
                            yield f"data: {json.dumps(msg)}\n\n"
                            if msg["type"] in ("error", "complete"):
                                break
                        except queue.Empty:
                            break
                    break
                continue

            yield f"data: {json.dumps(msg)}\n\n"
            if msg["type"] in ("error", "complete"):
                break

        thread.join()

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/results")
@log_errors
def get_dataset_results(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    manager: DepStateManager = None,  # pyright: ignore[reportArgumentType]
) -> DatasetSearchPage:
    """Get paginated results from the last dataset search.

    Args:
        page: Page number (1-indexed)
        page_size: Results per page (1-100)

    Returns:
        Paginated results with metadata
    """
    search_state = manager.state.dataset_search_state
    if search_state is None:
        raise HTTPException(
            status_code=404,
            detail="No search results available. Perform a search first.",
        )

    total_results = len(search_state.results)
    total_pages = max(1, (total_results + page_size - 1) // page_size)

    if page > total_pages and total_results > 0:
        raise HTTPException(
            status_code=400,
            detail=f"Page {page} exceeds total pages {total_pages}",
        )

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_results = search_state.results[start_idx:end_idx]

    return DatasetSearchPage(
        results=[DatasetSearchResult(**r) for r in page_results],
        page=page,
        page_size=page_size,
        total_results=total_results,
        total_pages=total_pages,
    )


@router.get("/metadata")
@log_errors
def get_dataset_metadata(
    manager: DepStateManager = None,  # pyright: ignore[reportArgumentType]
) -> DatasetSearchMetadata:
    """Get metadata from the last dataset search.

    Returns:
        Search metadata (query, split, total results, search time)
    """
    search_state = manager.state.dataset_search_state
    if search_state is None:
        raise HTTPException(
            status_code=404,
            detail="No search results available. Perform a search first.",
        )

    return DatasetSearchMetadata(**search_state.metadata)
