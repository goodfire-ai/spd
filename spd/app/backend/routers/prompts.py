"""Prompt listing endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from spd.app.backend.dependencies import DepLoadedRun, DepStateManager
from spd.app.backend.utils import log_errors

# =============================================================================
# Schemas
# =============================================================================


class PromptPreview(BaseModel):
    """Preview of a stored prompt for listing."""

    id: int
    token_ids: list[int]
    tokens: list[str]
    preview: str


router = APIRouter(prefix="/api/prompts", tags=["prompts"])


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


@router.post("/custom")
@log_errors
def create_custom_prompt(
    text: str,
    manager: DepStateManager,
    loaded: DepLoadedRun,
) -> PromptPreview:
    """Create a custom prompt from text and store it.

    Returns the created prompt with its ID for further operations.
    """
    db = manager.db

    token_ids = loaded.tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise HTTPException(status_code=400, detail="Text produced no tokens")

    prompt_id = db.add_custom_prompt(
        run_id=loaded.run.id,
        token_ids=token_ids,
        context_length=loaded.context_length,
    )

    token_strings = [loaded.token_strings[t] for t in token_ids]
    return PromptPreview(
        id=prompt_id,
        token_ids=token_ids,
        tokens=token_strings,
        preview="".join(token_strings[:10]) + ("..." if len(token_strings) > 10 else ""),
    )
