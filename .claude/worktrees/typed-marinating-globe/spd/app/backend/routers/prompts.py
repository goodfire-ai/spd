"""Prompt listing and generation endpoints."""

import json
from collections.abc import Generator
from typing import Annotated

import torch
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from spd.app.backend.compute import compute_ci_only, extract_active_from_ci
from spd.app.backend.dependencies import DepLoadedRun, DepStateManager
from spd.app.backend.utils import log_errors
from spd.configs import LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import extract_batch_data

# =============================================================================
# Schemas
# =============================================================================


class PromptPreview(BaseModel):
    """Preview of a stored prompt for listing."""

    id: int
    token_ids: list[int]
    tokens: list[str]
    preview: str


class PromptSearchQuery(BaseModel):
    """Query parameters for prompt search."""

    components: list[str]
    mode: str


class PromptSearchResponse(BaseModel):
    """Response from prompt search endpoint."""

    query: PromptSearchQuery
    count: int
    results: list[PromptPreview]


router = APIRouter(prefix="/api/prompts", tags=["prompts"])

DEVICE = get_device()


@router.get("")
@log_errors
def list_prompts(manager: DepStateManager, loaded: DepLoadedRun) -> list[PromptPreview]:
    """Return list of all prompts for the loaded run with matching context length."""
    db = manager.db
    prompt_ids = db.get_all_prompt_ids(loaded.run.id, loaded.context_length)

    results: list[PromptPreview] = []
    for pid in prompt_ids:
        prompt = db.get_prompt(pid)
        assert prompt is not None, f"Prompt {pid} in index but not in DB"
        token_strings = [loaded.token_strings[t] for t in prompt.token_ids]
        results.append(
            PromptPreview(
                id=prompt.id,
                token_ids=prompt.token_ids,
                tokens=token_strings,
                preview="".join(token_strings[:10]) + ("..." if len(token_strings) > 10 else ""),
            )
        )
    return results


BATCH_SIZE = 32


@router.post("/generate")
@log_errors
def generate_prompts(
    n_prompts: int,
    manager: DepStateManager,
    loaded: DepLoadedRun,
) -> StreamingResponse:
    """Generate prompts from training data with CI harvesting.

    Streams progress updates and stores prompts with their active components
    (for the inverted index used by search).
    """
    db = manager.db
    spd_config = loaded.config

    task_config = spd_config.task_config
    assert isinstance(task_config, LMTaskConfig)
    train_data_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=spd_config.tokenizer_name,
        split=task_config.train_data_split,
        n_ctx=loaded.context_length,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=task_config.shuffle_each_epoch,
    )
    logger.info(f"[API] Creating train loader for run {loaded.run.wandb_path}")
    train_loader, _ = create_data_loader(
        dataset_config=train_data_config,
        batch_size=BATCH_SIZE,
        buffer_size=task_config.buffer_size,
        global_seed=spd_config.seed,
    )

    def generate() -> Generator[str]:
        added_count = 0

        for batch in train_loader:
            if added_count >= n_prompts:
                break

            tokens = extract_batch_data(batch).to(DEVICE)
            batch_size, n_seq = tokens.shape

            # Compute CI for the whole batch
            ci_result = compute_ci_only(
                model=loaded.model,
                tokens=tokens,
                sampling=loaded.config.sampling,
            )

            # Process each sequence in the batch
            prompts = []
            for i in range(batch_size):
                if i % 5 == 0:
                    progress = min(added_count / n_prompts, 1.0)
                    progress_data = {"type": "progress", "progress": progress, "count": added_count}
                    yield f"data: {json.dumps(progress_data)}\n\n"

                if added_count >= n_prompts:
                    break

                token_ids = tokens[i].tolist()

                # Slice CI for this single sequence
                ci_single = {k: v[i : i + 1] for k, v in ci_result.ci_lower_leaky.items()}
                target_out_probs_single = ci_result.target_out_probs[i : i + 1]

                # Extract active components for inverted index
                active_components = extract_active_from_ci(
                    ci_lower_leaky=ci_single,
                    target_out_probs=target_out_probs_single,
                    ci_threshold=0.0,
                    output_prob_threshold=0.01,
                    n_seq=n_seq,
                )

                # Add to DB with active components
                prompts.append((token_ids, active_components))
                added_count += 1

            db.add_prompts(loaded.run.id, prompts, loaded.context_length)

        # Final result
        total = db.get_prompt_count(loaded.run.id, loaded.context_length)
        complete_data = {
            "type": "complete",
            "prompts_added": added_count,
            "total_prompts": total,
        }
        yield f"data: {json.dumps(complete_data)}\n\n"
        logger.info(f"[API] Generated {added_count} prompts for run {loaded.run.id}")

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/search")
@log_errors
def search_prompts(
    manager: DepStateManager,
    loaded: DepLoadedRun,
    components: str = "",
    mode: Annotated[str, Query(pattern="^(all|any)$")] = "all",
) -> PromptSearchResponse:
    """Search for prompts with specified components in the loaded run."""
    db = manager.db

    component_list = [c.strip() for c in components.split(",") if c.strip()]
    if not component_list:
        raise HTTPException(status_code=400, detail="No components specified")

    require_all = mode == "all"
    prompt_ids = db.find_prompts_with_components(
        loaded.run.id, component_list, require_all=require_all
    )

    results: list[PromptPreview] = []
    for pid in prompt_ids:
        prompt = db.get_prompt(pid)
        assert prompt is not None, f"Prompt {pid} in index but not in DB"
        token_strings = [loaded.token_strings[t] for t in prompt.token_ids]
        results.append(
            PromptPreview(
                id=prompt.id,
                token_ids=prompt.token_ids,
                tokens=token_strings,
                preview="".join(token_strings[:10]) + ("..." if len(token_strings) > 10 else ""),
            )
        )

    return PromptSearchResponse(
        query=PromptSearchQuery(components=component_list, mode=mode),
        count=len(results),
        results=results,
    )


@router.post("/custom")
@log_errors
def create_custom_prompt(
    text: str,
    manager: DepStateManager,
    loaded: DepLoadedRun,
) -> PromptPreview:
    """Create a custom prompt from text, computing CI and storing it.

    Returns the created prompt with its ID for further operations.
    """
    db = manager.db

    # Tokenize
    token_ids = loaded.tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise HTTPException(status_code=400, detail="Text produced no tokens")

    n_seq = len(token_ids)
    tokens_tensor = torch.tensor([token_ids], device=DEVICE)

    # Compute CI
    ci_result = compute_ci_only(
        model=loaded.model,
        tokens=tokens_tensor,
        sampling=loaded.config.sampling,
    )

    # Extract active components for inverted index
    active_components = extract_active_from_ci(
        ci_lower_leaky=ci_result.ci_lower_leaky,
        target_out_probs=ci_result.target_out_probs,
        ci_threshold=0.0,
        output_prob_threshold=0.01,
        n_seq=n_seq,
    )

    # Save to DB
    prompt_id = db.add_custom_prompt(
        run_id=loaded.run.id,
        token_ids=token_ids,
        active_components=active_components,
        context_length=loaded.context_length,
    )

    token_strings = [loaded.token_strings[t] for t in token_ids]
    return PromptPreview(
        id=prompt_id,
        token_ids=token_ids,
        tokens=token_strings,
        preview="".join(token_strings[:10]) + ("..." if len(token_strings) > 10 else ""),
    )
