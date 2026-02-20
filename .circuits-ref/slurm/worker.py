"""SLURM array task worker for processing attribution graph prompts.

This script is executed by each SLURM array task. It uses a two-phase approach:

Phase 1 (GPU-bound): For each prompt sequentially:
  - Generate attribution graph (RelP)
  - Label neurons from database
  - Cluster neurons
  - Save intermediate results

Phase 2 (IO-bound): Batch async LLM analysis:
  - Load all cluster results
  - Run LLM analysis concurrently with rate limiting
  - Save final outputs

This approach maximizes GPU utilization during Phase 1, then efficiently
batches API calls during Phase 2.

Usage:
    python -m slurm.worker --job-dir outputs/jobs/job_20250112 --task-id 0
"""

import argparse
import asyncio
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables from .env file (for API keys)
# Use explicit path since cwd might differ on SLURM nodes
_repo_root = Path(__file__).parent.parent
load_dotenv(_repo_root / ".env")

from .manifest import JobManifest, PromptStatus

# Add parent to path for circuits imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from circuits.async_analysis import RateLimiter, analyze_modules_async
from circuits.labeling import label_graph
from circuits.pipeline import (
    PipelineConfig,
    apply_chat_template,
    cluster_graph,
    format_for_viewer,
    slugify,
)
from circuits.relp import RelPAttributor, RelPConfig


@dataclass
class IntermediateResult:
    """Stores intermediate results between Phase 1 and Phase 2."""
    prompt_status: PromptStatus
    graph: dict[str, Any]
    clusters: dict[str, Any]
    slug: str
    output_dir: Path


def load_model(config: PipelineConfig):
    """Load the model and tokenizer."""
    print(f"Loading model: {config.model_name}")
    start = time.time()

    dtype = getattr(torch, config.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        device_map=config.device,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.1f}s")

    return model, tokenizer


def process_prompt_phase1(
    prompt_status: PromptStatus,
    config: PipelineConfig,
    model,
    tokenizer,
    output_dir: Path,
) -> IntermediateResult:
    """Phase 1: GPU-bound processing (RelP, labeling, clustering).

    Does NOT run LLM analysis - that's batched in Phase 2.
    """
    prompt = prompt_status.prompt
    answer_prefix = prompt_status.answer_prefix or config.answer_prefix
    slug = f"relp-{slugify(prompt)}"

    # Apply chat template
    if config.use_chat_template:
        formatted_prompt = apply_chat_template(tokenizer, prompt, answer_prefix)
    else:
        formatted_prompt = prompt

    # Generate attribution graph
    relp_config = RelPConfig(k=config.k, tau=config.tau, use_neuron_labels=False)
    attributor = RelPAttributor(model, tokenizer, config=relp_config)
    graph = attributor.compute_attributions(formatted_prompt)
    attributor.cleanup()

    # Store original prompt
    graph["metadata"]["prompt"] = prompt

    # Format for viewer
    graph = format_for_viewer(graph, slug)

    # Label neurons from database
    if config.label_neurons:
        graph = label_graph(graph, verbose=False)

    # Save graph
    if config.save_intermediate:
        graph_path = output_dir / f"{slug}-graph.json"
        with open(graph_path, "w") as f:
            json.dump(graph, f, indent=2)

    # Cluster neurons
    clusters = cluster_graph(
        graph,
        skip_special_tokens=config.skip_special_tokens,
        min_cluster_size=config.min_cluster_size,
        max_depth=config.max_cluster_depth,
        verbose=False,
    )

    # Save clusters
    if config.save_intermediate:
        clusters_path = output_dir / f"{slug}-clusters.json"
        with open(clusters_path, "w") as f:
            json.dump(clusters, f, indent=2)

    return IntermediateResult(
        prompt_status=prompt_status,
        graph=graph,
        clusters=clusters,
        slug=slug,
        output_dir=output_dir,
    )


async def process_phase2_async(
    results: list[IntermediateResult],
    config: PipelineConfig,
    rate_limiter: RateLimiter,
) -> list[tuple[IntermediateResult, dict[str, Any]]]:
    """Phase 2: Batch async LLM analysis for all results."""

    async def analyze_one(result: IntermediateResult) -> tuple[IntermediateResult, dict[str, Any]]:
        """Analyze a single result."""
        analysis = await analyze_modules_async(
            result.clusters,
            graph=result.graph,
            model=config.llm_model,
            provider=config.llm_provider,
            rate_limiter=rate_limiter,
            functional_split=config.functional_split,
            functional_split_min_size=config.functional_split_min_size,
            use_prompt_answer_split=config.use_prompt_answer_split,
            use_position_split=config.use_position_split,
            max_position_gap=config.max_position_gap,
            use_layer_split=config.use_layer_split,
            use_semantic_split=config.use_semantic_split,
            use_llm_split=config.use_llm_split,
            verbose=False,
        )
        return result, analysis

    # Run all analyses concurrently (rate limiter handles throttling)
    tasks = [analyze_one(r) for r in results]
    return await asyncio.gather(*tasks)


def run_worker(job_dir: Path, task_id: int):
    """Main worker execution loop with two-phase processing."""
    print("=== Worker Starting ===")
    print(f"Job directory: {job_dir}")
    print(f"Task ID: {task_id}")

    # Load manifest
    manifest = JobManifest(job_dir)
    if not manifest.exists():
        print(f"ERROR: Manifest not found in {job_dir}", file=sys.stderr)
        sys.exit(1)

    # Load config
    config_path = job_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config_dict = yaml.safe_load(f).get("config", {})
        config = PipelineConfig.from_dict(config_dict)
    else:
        config = PipelineConfig()

    # Get prompts for this task (supports resume)
    pending = manifest.get_pending_for_task(task_id)
    total_for_task = len(manifest.get_all_for_task(task_id))

    print(f"Prompts assigned to this task: {total_for_task}")
    print(f"Prompts to process (pending/failed): {len(pending)}")

    if not pending:
        print("No prompts to process, exiting")
        return

    # Output directory for this task
    output_dir = job_dir / "arrays" / f"task_{task_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model once
    model, tokenizer = load_model(config)

    # =========================================================================
    # PHASE 1: GPU-bound processing (sequential)
    # =========================================================================
    print(f"\n=== Phase 1: GPU Processing ({len(pending)} prompts) ===")
    phase1_start = time.time()

    intermediate_results: list[IntermediateResult] = []
    phase1_failed = 0

    for i, prompt_status in enumerate(pending):
        prompt_preview = (
            prompt_status.prompt[:50] + "..."
            if len(prompt_status.prompt) > 50
            else prompt_status.prompt
        )
        print(f"\n[{i+1}/{len(pending)}] Processing: {prompt_preview}")

        # Mark as running
        manifest.update_status(prompt_status.idx, "running")

        try:
            prompt_start = time.time()
            result = process_prompt_phase1(
                prompt_status,
                config,
                model,
                tokenizer,
                output_dir,
            )
            prompt_elapsed = time.time() - prompt_start
            intermediate_results.append(result)
            print(f"  Phase 1 done in {prompt_elapsed:.1f}s (graph: {len(result.graph['nodes'])} nodes, {result.clusters['methods'][0]['n_clusters']} clusters)")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"  FAILED: {error_msg}", file=sys.stderr)
            traceback.print_exc()
            manifest.update_status(prompt_status.idx, "failed", error=error_msg)
            phase1_failed += 1

    phase1_elapsed = time.time() - phase1_start
    print(f"\nPhase 1 complete: {len(intermediate_results)} succeeded, {phase1_failed} failed in {phase1_elapsed:.1f}s")

    # Free GPU memory before Phase 2
    del model
    del tokenizer
    torch.cuda.empty_cache()

    # =========================================================================
    # PHASE 2: LLM Analysis (async batch)
    # =========================================================================
    if config.run_llm_analysis and intermediate_results:
        print(f"\n=== Phase 2: Async LLM Analysis ({len(intermediate_results)} graphs) ===")
        phase2_start = time.time()

        # Get distributed rate limit (total / n_workers)
        worker_rate_limit = manifest.get_worker_rate_limit()
        llm_concurrency = config_dict.get("llm_concurrency", 64)

        # Burst size is min of concurrency and rate limit (can't burst more than rate allows)
        burst_size = min(llm_concurrency, max(1, worker_rate_limit // 6))  # ~10s worth of burst

        print(f"  Rate limit: {worker_rate_limit}/min (distributed), concurrency: {llm_concurrency}, burst: {burst_size}")
        rate_limiter = RateLimiter(requests_per_minute=worker_rate_limit, burst_size=burst_size)

        # Run async analysis
        try:
            analyzed_results = asyncio.run(
                process_phase2_async(intermediate_results, config, rate_limiter)
            )

            # Save final outputs
            for result, analysis in analyzed_results:
                final_path = result.output_dir / f"{result.slug}-analysis.json"
                with open(final_path, "w") as f:
                    json.dump(analysis, f, indent=2)

                # Mark as completed
                manifest.update_status(
                    result.prompt_status.idx,
                    "completed",
                    output_path=str(final_path),
                )
                print(f"  Saved: {final_path.name}")

            phase2_elapsed = time.time() - phase2_start
            print(f"\nPhase 2 complete: {len(analyzed_results)} analyses in {phase2_elapsed:.1f}s")

        except Exception as e:
            error_msg = f"Phase 2 batch error: {type(e).__name__}: {str(e)}"
            print(f"  FAILED: {error_msg}", file=sys.stderr)
            traceback.print_exc()

            # Mark all pending as failed
            for result in intermediate_results:
                manifest.update_status(
                    result.prompt_status.idx,
                    "failed",
                    error=error_msg,
                )

    else:
        # No LLM analysis - just save cluster results as final output
        print("\n=== Skipping Phase 2 (LLM analysis disabled) ===")
        for result in intermediate_results:
            # Save analysis without LLM synthesis
            from circuits.analysis import compute_module_flow

            method_data = result.clusters["methods"][0]
            analysis = compute_module_flow(result.graph, method_data, result.clusters)

            final_path = result.output_dir / f"{result.slug}-analysis.json"
            with open(final_path, "w") as f:
                json.dump(analysis, f, indent=2)

            manifest.update_status(
                result.prompt_status.idx,
                "completed",
                output_path=str(final_path),
            )

    # =========================================================================
    # Summary
    # =========================================================================
    progress = manifest.get_progress()
    completed = len([r for r in manifest.get_all_for_task(task_id) if manifest._read()["prompts"][r.idx]["status"] == "completed"])
    failed = len(pending) - len(intermediate_results) + (len(intermediate_results) - completed if config.run_llm_analysis else 0)

    total_elapsed = time.time() - phase1_start
    print("\n=== Worker Complete ===")
    print(f"Processed: {len(pending)} prompts")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_elapsed:.1f}s")
    if completed > 0:
        print(f"Avg time per prompt: {total_elapsed / len(pending):.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="SLURM worker for attribution graph generation"
    )
    parser.add_argument(
        "--job-dir",
        type=Path,
        required=True,
        help="Path to the job directory",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        required=True,
        help="SLURM array task ID",
    )

    args = parser.parse_args()
    run_worker(args.job_dir, args.task_id)


if __name__ == "__main__":
    main()
