"""Run management endpoints."""

from urllib.parse import unquote

import torch
import yaml
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from spd.app.backend.compute import get_sources_by_target
from spd.app.backend.dependencies import DepStateManager
from spd.app.backend.schemas import LoadedRun
from spd.app.backend.state import RunState
from spd.app.backend.utils import build_token_lookup, log_errors, validate_wandb_path
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device

router = APIRouter(prefix="/api", tags=["runs"])

DEVICE = get_device()


@router.post("/runs/load")
@log_errors
def load_run(wandb_path: str, context_length: int, manager: DepStateManager):
    """Load a run by its wandb path. Creates the run in DB if not found.

    Expects path in normalized format: entity/project/runId
    (Frontend handles parsing various formats into this normalized form)

    This loads the model onto GPU and makes it available for attribution computation.
    """
    db = manager.db

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

    run = db.get_run_by_wandb_path(full_wandb_path)
    if run is None:
        new_run_id = db.create_run(full_wandb_path)
        run = db.get_run(new_run_id)
        assert run is not None
        logger.info(f"[API] Created new run in DB: {run.id}")
    else:
        logger.info(f"[API] Found existing run in DB: {run.id}")

    # If already loaded, skip model load
    if manager.run_state is not None and manager.run_state.run.id == run.id:
        logger.info(f"[API] Run {run.id} already loaded, skipping")
        return {"status": "already_loaded", "run_id": run.id, "wandb_path": run.wandb_path}

    # Unload previous run if any
    if manager.run_state is not None:
        logger.info(f"[API] Unloading previous run {manager.run_state.run.id}")
        del manager.run_state.model
        torch.cuda.empty_cache()
        manager.run_state = None

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
    token_strings = build_token_lookup(loaded_tokenizer, spd_config.tokenizer_name)

    task_config = spd_config.task_config
    assert isinstance(task_config, LMTaskConfig)
    train_data_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=spd_config.tokenizer_name,
        split=task_config.train_data_split,
        n_ctx=context_length,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=task_config.shuffle_each_epoch,
    )
    train_loader, _ = create_data_loader(
        dataset_config=train_data_config,
        batch_size=32,
        buffer_size=task_config.buffer_size,
        global_seed=spd_config.seed,
    )
    manager.run_state = RunState(
        run=run,
        model=model,
        tokenizer=loaded_tokenizer,
        sources_by_target=sources_by_target,
        config=spd_config,
        token_strings=token_strings,
        train_loader=train_loader,
        context_length=context_length,
    )

    logger.info(f"[API] Run {run.id} loaded on {DEVICE}")
    return {"status": "loaded", "run_id": run.id, "wandb_path": run.wandb_path}


@router.get("/status")
@log_errors
def get_status(manager: DepStateManager) -> LoadedRun | None:
    """Get current server status."""
    if manager.run_state is None:
        return None

    run = manager.run_state.run
    config_yaml = yaml.dump(
        manager.run_state.config.model_dump(), default_flow_style=False, sort_keys=False
    )

    context_length = manager.run_state.context_length

    prompt_count = manager.db.get_prompt_count(run.id, context_length)

    return LoadedRun(
        id=run.id,
        wandb_path=run.wandb_path,
        config_yaml=config_yaml,
        has_activation_contexts=manager.db.has_activation_contexts(run.id),
        has_prompts=prompt_count > 0,
        prompt_count=prompt_count,
    )


@router.get("/health")
@log_errors
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
