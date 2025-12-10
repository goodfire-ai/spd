"""One-off script to harvest component correlations for a run.

Usage:
    python -m spd.app.scripts.harvest_correlations <wandb_path> [options]

Example:
    python -m spd.app.scripts.harvest_correlations anthropic/spd/abc123 --n_batches 500
"""

import json
import traceback
from pathlib import Path
from typing import Any

import fire
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from spd.app.backend.lib.component_correlations import (
    get_correlations_path,
    get_token_stats_path,
    harvest_correlations,
)
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.wandb_utils import parse_wandb_run_path


def _update_status(
    status_file: Path,
    status: str,
    error: str | None = None,
    n_tokens: int | None = None,
    n_components: int | None = None,
) -> None:
    """Update status in the status file, preserving job_id, submitted_at, params."""
    assert status_file.exists(), f"status file must exist: {status_file}"
    existing = json.loads(status_file.read_text())

    # Preserve fields from initial submission
    assert "job_id" in existing, "status file missing job_id"
    assert "submitted_at" in existing, "status file missing submitted_at"
    assert "params" in existing, "status file missing params"

    data: dict[str, Any] = {
        "status": status,
        "job_id": existing["job_id"],
        "submitted_at": existing["submitted_at"],
        "params": existing["params"],
    }
    if error is not None:
        data["error"] = error
    if n_tokens is not None:
        data["n_tokens"] = n_tokens
    if n_components is not None:
        data["n_components"] = n_components

    status_file.write_text(json.dumps(data, indent=2))


def main(
    wandb_path: str,
    n_batches: int = 500,
    batch_size: int = 32,
    context_length: int = 128,
    ci_threshold: float = 1e-6,
    status_file: str | None = None,
) -> None:
    """Harvest component correlations for a run.

    Args:
        wandb_path: W&B run path (entity/project/run_id)
        n_batches: Number of batches to process
        batch_size: Batch size
        context_length: Context length
        ci_threshold: CI threshold for binarization
        status_file: Path to status file (created by job submission, required for job tracking)
    """
    status_path = Path(status_file) if status_file else None

    # Update status to running
    if status_path:
        _update_status(status_path, "running")

    try:
        device = get_device()
        logger.info(f"Using device: {device}")

        # Parse and normalize wandb path
        entity, project, run_id = parse_wandb_run_path(wandb_path)
        clean_wandb_path = f"{entity}/{project}/{run_id}"
        logger.info(f"Loading run: {clean_wandb_path}")

        # Load model
        run_info = SPDRunInfo.from_path(clean_wandb_path)
        model = ComponentModel.from_run_info(run_info)
        model = model.to(device)
        model.eval()

        # Load config
        spd_config = run_info.config
        assert spd_config.tokenizer_name is not None

        # Load tokenizer (needed for dataset)
        logger.info(f"Loading tokenizer: {spd_config.tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(spd_config.tokenizer_name)
        assert isinstance(tokenizer, PreTrainedTokenizerFast)

        # Create data loader
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
            batch_size=batch_size,
            buffer_size=task_config.buffer_size,
            global_seed=spd_config.seed,
        )

        # Get vocab size from tokenizer
        vocab_size = tokenizer.vocab_size
        assert vocab_size is not None, "tokenizer.vocab_size is None"

        # Harvest correlations and token stats
        logger.info(
            f"Harvesting correlations + token stats: {n_batches} batches × {batch_size} batch_size "
            f"× {context_length} context_length = {n_batches * batch_size * context_length:,} tokens"
        )

        result = harvest_correlations(
            config=spd_config,
            cm=model,
            train_loader=train_loader,
            ci_threshold=ci_threshold,
            n_batches=n_batches,
            vocab_size=vocab_size,
        )

        # Save correlations
        correlations_path = get_correlations_path(run_id)
        result.correlations.save(correlations_path)

        # Save token stats
        token_stats_path = get_token_stats_path(run_id)
        result.token_stats.save(token_stats_path)

        logger.info("Done!")
        logger.info(f"  - Correlations saved to {correlations_path}")
        logger.info(f"  - Token stats saved to {token_stats_path}")
        logger.info(f"  - Components: {len(result.correlations.component_keys)}")
        logger.info(f"  - Tokens processed: {result.correlations.n_tokens:,}")

        # Quick sanity check: show top correlations for first active component
        active_components = [
            k
            for i, k in enumerate(result.correlations.component_keys)
            if result.correlations.count_i[i] > 0
        ]
        if active_components:
            test_key = active_components[0]
            top_corr = result.correlations.get_correlated(test_key, metric="f1", top_k=5)
            logger.info(f"  - Sample correlations for {test_key}:")
            for c in top_corr:
                logger.info(f"      {c.component_key}: F1={c.score:.4f}")

            # Show sample input token stats
            input_stats = result.token_stats.get_input_stats(test_key, tokenizer, top_k=5)
            if input_stats:
                logger.info(f"  - Sample input token PMI for {test_key}:")
                for tok, pmi_val in input_stats.top_pmi[:5]:
                    logger.info(f"      {tok!r}: PMI={pmi_val:.2f}")

            # Show sample output token stats
            output_stats = result.token_stats.get_output_stats(test_key, tokenizer, top_k=5)
            if output_stats:
                logger.info(f"  - Sample output token PMI for {test_key}:")
                for tok, pmi_val in output_stats.top_pmi[:5]:
                    logger.info(f"      {tok!r}: PMI={pmi_val:.2f}")

        # Update status to completed
        if status_path:
            _update_status(
                status_path,
                "completed",
                n_tokens=result.correlations.n_tokens,
                n_components=len(result.correlations.component_keys),
            )

    except Exception as e:
        logger.error(f"Harvest failed: {e}")
        logger.error(traceback.format_exc())
        if status_path:
            _update_status(status_path, "failed", error=str(e))
        raise


if __name__ == "__main__":
    fire.Fire(main)
