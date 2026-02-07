"""Dataset search endpoints for SimpleStories exploration.

This module provides search functionality for the SimpleStories dataset,
independent of any loaded SPD run. Results are cached in memory for pagination.
"""

import random
import time
from typing import Annotated, Any

import torch
from datasets import Dataset, load_dataset
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from spd.app.backend.dependencies import DepLoadedRun, DepStateManager
from spd.app.backend.state import DatasetSearchState
from spd.app.backend.utils import log_errors
from spd.log import logger
from spd.utils.distributed_utils import get_device

# =============================================================================
# Schemas
# =============================================================================


class DatasetSearchResult(BaseModel):
    """A single search result from the SimpleStories dataset."""

    story: str
    occurrence_count: int
    topic: str | None = None
    theme: str | None = None


class TokenizedSearchResult(BaseModel):
    """A tokenized search result with per-token probability."""

    tokens: list[str]
    next_token_probs: list[float | None]
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


class TokenizedSearchPage(BaseModel):
    """Paginated tokenized results from a dataset search."""

    results: list[TokenizedSearchResult]
    query: str
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


@router.get("/results_tokenized")
@log_errors
def get_tokenized_results(
    loaded: DepLoadedRun,
    manager: DepStateManager,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=20)] = 10,
    max_tokens: Annotated[int, Query(ge=16, le=512)] = 256,
) -> TokenizedSearchPage:
    """Get paginated tokenized results with per-token probability.

    Requires a loaded run for model inference. Results are tokenized and
    run through the model to compute next-token probabilities.

    Args:
        page: Page number (1-indexed)
        page_size: Results per page (1-20, lower limit due to model inference)
        max_tokens: Maximum tokens per result (truncated if longer)

    Returns:
        Paginated tokenized results with probabilities
    """
    search_state = manager.state.dataset_search_state
    if search_state is None:
        raise HTTPException(
            status_code=404,
            detail="No search results available. Perform a search first.",
        )

    device = get_device()
    model = loaded.model
    tokenizer = loaded.tokenizer

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

    tokenized_results: list[TokenizedSearchResult] = []

    for result in page_results:
        story: str = result["story"]

        token_ids = tokenizer.encode(story, add_special_tokens=False)
        if len(token_ids) > max_tokens:
            token_ids = token_ids[:max_tokens]

        if len(token_ids) == 0:
            continue

        tokens_tensor = torch.tensor([token_ids], device=device)

        with torch.no_grad():
            logits = model(tokens_tensor)
            probs = torch.softmax(logits, dim=-1)

        next_token_probs: list[float | None] = []
        for i in range(len(token_ids) - 1):
            next_token_id = token_ids[i + 1]
            prob = probs[0, i, next_token_id].item()
            next_token_probs.append(prob)
        next_token_probs.append(None)

        token_strings = [loaded.token_strings[t] for t in token_ids]

        tokenized_results.append(
            TokenizedSearchResult(
                tokens=token_strings,
                next_token_probs=next_token_probs,
                occurrence_count=result["occurrence_count"],
                topic=result.get("topic"),
                theme=result.get("theme"),
            )
        )

    query = search_state.metadata.get("query", "")

    return TokenizedSearchPage(
        results=tokenized_results,
        query=query,
        page=page,
        page_size=page_size,
        total_results=total_results,
        total_pages=total_pages,
    )


class RandomSamplesResult(BaseModel):
    """Random samples from the dataset."""

    results: list[DatasetSearchResult]
    total_available: int
    seed: int


@router.get("/random")
@log_errors
def get_random_samples(
    n_samples: Annotated[int, Query(ge=1, le=200)] = 100,
    seed: Annotated[int, Query(ge=0)] = 42,
    split: Annotated[str, Query(pattern="^(train|test)$")] = "train",
) -> RandomSamplesResult:
    """Get random samples from the SimpleStories dataset.

    Args:
        n_samples: Number of random samples to return (1-200)
        seed: Random seed for reproducibility
        split: Dataset split ("train" or "test")

    Returns:
        Random samples with metadata
    """
    logger.info(f"Loading SimpleStories dataset (split={split}) for random sampling...")
    dataset = load_dataset("lennart-finke/SimpleStories", split=split)
    assert isinstance(dataset, Dataset), f"Expected Dataset, got {type(dataset)}"

    total_available = len(dataset)
    actual_samples = min(n_samples, total_available)

    # Generate random indices directly instead of shuffling entire dataset (~100x faster)
    rng = random.Random(seed)
    indices = rng.sample(range(total_available), actual_samples)
    samples = dataset.select(indices)

    results = []
    for item in samples:
        item_dict: dict[str, Any] = dict(item)
        results.append(
            DatasetSearchResult(
                story=item_dict["story"],
                occurrence_count=0,
                topic=item_dict.get("topic"),
                theme=item_dict.get("theme"),
            )
        )

    logger.info(f"Returned {len(results)} random samples from {total_available} total stories")

    return RandomSamplesResult(
        results=results,
        total_available=total_available,
        seed=seed,
    )


class TokenizedSample(BaseModel):
    """A single tokenized sample with per-token next-token probability."""

    tokens: list[str]
    next_token_probs: list[float | None]  # Probability of next token; None for last position
    topic: str | None = None
    theme: str | None = None


class RandomSamplesWithLossResult(BaseModel):
    """Random samples with tokenized data and next-token probabilities."""

    results: list[TokenizedSample]
    total_available: int
    seed: int


@router.get("/random_with_loss")
@log_errors
def get_random_samples_with_loss(
    loaded: DepLoadedRun,
    n_samples: Annotated[int, Query(ge=1, le=50)] = 20,
    seed: Annotated[int, Query(ge=0)] = 42,
    split: Annotated[str, Query(pattern="^(train|test)$")] = "train",
    max_tokens: Annotated[int, Query(ge=16, le=512)] = 256,
) -> RandomSamplesWithLossResult:
    """Get random samples with tokenized data and per-token next-token probability.

    This endpoint requires a loaded run (for model and tokenizer).
    Each sample is tokenized and run through the model to compute probabilities.

    Args:
        n_samples: Number of random samples to return (1-50, lower limit due to model inference)
        seed: Random seed for reproducibility
        split: Dataset split ("train" or "test")
        max_tokens: Maximum tokens per sample (truncated if longer)

    Returns:
        Tokenized samples with next-token probability per token
    """
    device = get_device()
    model = loaded.model
    tokenizer = loaded.tokenizer

    logger.info(f"Loading SimpleStories dataset (split={split}) for random sampling with loss...")
    dataset = load_dataset("lennart-finke/SimpleStories", split=split)
    assert isinstance(dataset, Dataset), f"Expected Dataset, got {type(dataset)}"

    total_available = len(dataset)
    actual_samples = min(n_samples, total_available)

    rng = random.Random(seed)
    indices = rng.sample(range(total_available), actual_samples)
    samples = dataset.select(indices)

    results: list[TokenizedSample] = []

    for item in samples:
        item_dict: dict[str, Any] = dict(item)
        story: str = item_dict["story"]

        token_ids = tokenizer.encode(story, add_special_tokens=False)
        if len(token_ids) > max_tokens:
            token_ids = token_ids[:max_tokens]

        if len(token_ids) == 0:
            continue

        tokens_tensor = torch.tensor([token_ids], device=device)

        with torch.no_grad():
            logits = model(tokens_tensor)
            probs = torch.softmax(logits, dim=-1)

        # Get probability of next token at each position
        next_token_probs: list[float | None] = []
        for i in range(len(token_ids) - 1):
            next_token_id = token_ids[i + 1]
            prob = probs[0, i, next_token_id].item()
            next_token_probs.append(prob)
        next_token_probs.append(None)  # No next token for last position

        token_strings = [loaded.token_strings[t] for t in token_ids]

        results.append(
            TokenizedSample(
                tokens=token_strings,
                next_token_probs=next_token_probs,
                topic=item_dict.get("topic"),
                theme=item_dict.get("theme"),
            )
        )

    logger.info(
        f"Returned {len(results)} tokenized samples with CE loss from {total_available} total stories"
    )

    return RandomSamplesWithLossResult(
        results=results,
        total_available=total_available,
        seed=seed,
    )
