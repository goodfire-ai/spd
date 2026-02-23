"""Dataset search endpoints for SimpleStories exploration.

This module provides search functionality for the SimpleStories dataset,
independent of any loaded SPD run. Results are cached in memory for pagination.
"""

import time
from typing import Annotated, Any

from datasets import Dataset, load_dataset
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from spd.app.backend.dependencies import DepStateManager
from spd.app.backend.state import DatasetSearchState
from spd.app.backend.utils import log_errors
from spd.log import logger

# =============================================================================
# Schemas
# =============================================================================


class DatasetSearchResult(BaseModel):
    """A single search result from the SimpleStories dataset."""

    story: str
    occurrence_count: int
    topic: str | None = None
    theme: str | None = None


class DatasetSearchMetadata(BaseModel):
    """Metadata about a completed dataset search."""

    query: str
    split: str
    total_results: int
    search_time_seconds: float


class DatasetSearchPage(BaseModel):
    """Paginated results from a dataset search."""

    results: list[DatasetSearchResult]
    page: int
    page_size: int
    total_results: int
    total_pages: int


router = APIRouter(prefix="/api/dataset", tags=["dataset"])


@router.post("/search")
@log_errors
def search_dataset(
    query: Annotated[str, Query(min_length=1)],
    manager: DepStateManager,
    split: Annotated[str, Query(pattern="^(train|test)$")] = "train",
) -> DatasetSearchMetadata:
    """Search SimpleStories dataset for stories containing query string.

    Caches results for pagination via /results endpoint.
    Works independently of any loaded run.

    Args:
        query: Text to search for (case-insensitive)
        split: Dataset split to search ("train" or "test")

    Returns:
        Search metadata (query, split, total results, search time)
    """
    start_time = time.time()
    search_query = query.lower()

    logger.info(f"Loading SimpleStories dataset (split={split})...")
    dataset = load_dataset("lennart-finke/SimpleStories", split=split)
    assert isinstance(dataset, Dataset), f"Expected Dataset, got {type(dataset)}"

    total_stories = len(dataset)
    logger.info(f"Searching {total_stories} stories for '{query}'...")

    filtered = dataset.filter(
        lambda x: search_query in x["story"].lower(),
        num_proc=8,
    )

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

    search_time = time.time() - start_time

    metadata = DatasetSearchMetadata(
        query=query,
        split=split,
        total_results=len(results),
        search_time_seconds=search_time,
    )
    manager.state.dataset_search_state = DatasetSearchState(
        results=results,
        metadata=metadata.model_dump(),
    )

    logger.info(
        f"Found {len(results)} results in {search_time:.2f}s (searched {total_stories} stories)"
    )

    return metadata


@router.get("/results")
@log_errors
def get_dataset_results(
    manager: DepStateManager,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
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
