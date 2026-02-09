"""Unified postprocessing pipeline for SPD runs.

All SLURM job scheduling lives here. Individual modules (harvest, attributions,
autointerp) only submit their own core jobs; eval orchestration is centralized.

Dependency graph:
    harvest (GPU array → merge)
    ├── intruder eval       (CPU, label-free)
    ├── interpret           (CPU, LLM calls)
    │   ├── detection       (CPU, label-dependent)
    │   └── fuzzing         (CPU, label-dependent)
    attributions (GPU array → merge, parallel with harvest)
"""

import secrets

from spd.log import logger
from spd.scripts.postprocess_config import PostprocessConfig
from spd.utils.git_utils import create_git_snapshot
from spd.utils.slurm import SlurmConfig, SubmitResult, generate_script, submit_slurm_job


def _submit_cpu_job(
    job_name: str,
    command: str,
    partition: str,
    time: str,
    snapshot_branch: str,
    dependency_job_id: str,
) -> SubmitResult:
    slurm_config = SlurmConfig(
        job_name=job_name,
        partition=partition,
        n_gpus=0,
        cpus_per_task=16,
        time=time,
        snapshot_branch=snapshot_branch,
        dependency_job_id=dependency_job_id,
    )
    script_content = generate_script(slurm_config, command)
    return submit_slurm_job(script_content, job_name)


def postprocess(wandb_path: str, config: PostprocessConfig) -> None:
    """Submit all postprocessing jobs with SLURM dependency chaining."""
    from spd.autointerp.scripts.run_slurm import submit_interpret
    from spd.dataset_attributions.scripts.run_slurm import submit_attributions
    from spd.harvest.scripts.run_slurm import harvest

    h = config.harvest
    total_gpus = h.n_gpus + (0 if config.attributions is None else config.attributions.n_gpus)
    assert total_gpus <= 8, f"Total GPUs ({total_gpus}) exceeds cluster limit of 8"

    run_id = f"postprocess-{secrets.token_hex(4)}"
    snapshot_branch, commit_hash = create_git_snapshot(run_id)
    logger.info(f"Created git snapshot: {snapshot_branch} ({commit_hash[:8]})")

    # === 1. Harvest (always) ===
    harvest_result = harvest(
        wandb_path=wandb_path,
        n_gpus=h.n_gpus,
        n_batches=h.n_batches,
        batch_size=h.batch_size,
        ci_threshold=h.ci_threshold,
        activation_examples_per_component=h.activation_examples_per_component,
        activation_context_tokens_per_side=h.activation_context_tokens_per_side,
        pmi_token_top_k=h.pmi_token_top_k,
        partition=h.partition,
        time=h.time,
        snapshot_branch=snapshot_branch,
    )

    # === 2. Attributions (parallel with harvest) ===
    if config.attributions is not None:
        a = config.attributions
        submit_attributions(
            wandb_path=wandb_path,
            n_gpus=a.n_gpus,
            n_batches=a.n_batches,
            batch_size=a.batch_size,
            ci_threshold=a.ci_threshold,
            partition=a.partition,
            time=a.time,
            snapshot_branch=snapshot_branch,
        )

    # === 3. Intruder eval (depends on harvest merge, label-free) ===
    if config.eval is not None:
        ev = config.eval
        intruder_cmd = " \\\n    ".join(
            [
                "python -m spd.autointerp.eval.scripts.run_intruder",
                f'"{wandb_path}"',
            ]
        )
        intruder_result = _submit_cpu_job(
            "spd-intruder-eval",
            intruder_cmd,
            ev.partition,
            ev.time,
            snapshot_branch,
            dependency_job_id=harvest_result.job_id,
        )
        logger.section("Intruder eval job submitted")
        logger.values(
            {
                "Job ID": intruder_result.job_id,
                "Depends on": f"harvest merge ({harvest_result.job_id})",
                "Log": intruder_result.log_pattern,
            }
        )

    # === 4. Interpret (depends on harvest merge) ===
    if config.interpret is not None:
        interp = config.interpret
        interpret_result, autointerp_run_id = submit_interpret(
            wandb_path=wandb_path,
            model=interp.model,
            limit=interp.limit,
            reasoning_effort=interp.reasoning_effort,
            config=None,
            partition=interp.partition,
            time=interp.time,
            cost_limit_usd=interp.cost_limit_usd,
            dependency_job_id=harvest_result.job_id,
            snapshot_branch=snapshot_branch,
        )

        # === 5. Detection scoring (depends on interpret) ===
        # === 6. Fuzzing scoring (depends on interpret) ===
        if config.eval is not None:
            ev = config.eval
            for scorer in ("detection", "fuzzing"):
                scoring_cmd = " \\\n    ".join(
                    [
                        "python -m spd.autointerp.scoring.scripts.run_label_scoring",
                        f'"{wandb_path}"',
                        f"--scorer {scorer}",
                        f"--autointerp_run_id {autointerp_run_id}",
                        f"--model {ev.eval_model}",
                    ]
                )
                scoring_result = _submit_cpu_job(
                    f"spd-{scorer}",
                    scoring_cmd,
                    ev.partition,
                    ev.time,
                    snapshot_branch,
                    dependency_job_id=interpret_result.job_id,
                )
                logger.section(f"{scorer.capitalize()} scoring job submitted")
                logger.values(
                    {
                        "Job ID": scoring_result.job_id,
                        "Depends on": f"interpret ({interpret_result.job_id})",
                        "Log": scoring_result.log_pattern,
                    }
                )
