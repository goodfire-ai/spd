"""Aggregate results from completed batch jobs.

Collects all output files, generates summary statistics, and optionally
creates an HTML index for browsing results.

Usage:
    # Generate summary
    python -m slurm.aggregator outputs/jobs/job_20250112_143052

    # With HTML index
    python -m slurm.aggregator outputs/jobs/job_20250112_143052 --create-index

    # Copy all outputs to merged directory
    python -m slurm.aggregator outputs/jobs/job_20250112_143052 --copy-outputs
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .manifest import JobManifest


def collect_outputs(manifest: JobManifest) -> list[dict[str, Any]]:
    """Collect all completed output files and their metadata."""
    outputs = []

    for output_path in manifest.get_completed_outputs():
        path = Path(output_path)
        if not path.exists():
            continue

        try:
            with open(path) as f:
                data = json.load(f)

            outputs.append({
                "path": str(path),
                "prompt": data.get("prompt", ""),
                "n_modules": data.get("n_modules", 0),
                "top_logits": data.get("top_logits", []),
            })
        except Exception as e:
            print(f"Warning: Failed to read {path}: {e}", file=sys.stderr)

    return outputs


def generate_summary(manifest: JobManifest) -> dict[str, Any]:
    """Generate summary statistics for a completed job."""
    progress = manifest.get_progress()
    outputs = collect_outputs(manifest)

    # Collect timing stats
    data = manifest._read()
    durations = [
        p["duration_seconds"]
        for p in data["prompts"]
        if p.get("duration_seconds")
    ]

    # Collect module stats
    module_counts = [o["n_modules"] for o in outputs if o.get("n_modules")]

    summary = {
        "job_id": progress["job_id"],
        "generated_at": datetime.now().isoformat(),
        "total_prompts": progress["total"],
        "completed": progress["completed"],
        "failed": progress["failed"],
        "completion_rate": progress["completed"] / progress["total"] if progress["total"] > 0 else 0,
        "timing": {
            "total_duration_seconds": sum(durations) if durations else 0,
            "avg_duration_seconds": sum(durations) / len(durations) if durations else 0,
            "min_duration_seconds": min(durations) if durations else 0,
            "max_duration_seconds": max(durations) if durations else 0,
        },
        "modules": {
            "avg_modules_per_graph": sum(module_counts) / len(module_counts) if module_counts else 0,
            "min_modules": min(module_counts) if module_counts else 0,
            "max_modules": max(module_counts) if module_counts else 0,
        },
        "outputs": outputs,
    }

    # Add failure details
    if progress["failed"] > 0:
        failed_prompts = manifest.get_failed_prompts()
        summary["failures"] = [
            {
                "idx": fp.idx,
                "prompt": fp.prompt[:100],
                "error": fp.error,
            }
            for fp in failed_prompts
        ]

    return summary


def generate_html_index(summary: dict[str, Any], output_path: Path) -> None:
    """Generate an HTML index page for browsing results."""
    outputs = summary.get("outputs", [])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Job {summary['job_id']} - Results</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #333; }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat {{
            display: inline-block;
            margin-right: 30px;
            padding: 10px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2563eb;
        }}
        .stat-label {{
            color: #666;
            font-size: 14px;
        }}
        .results {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .result {{
            padding: 15px 20px;
            border-bottom: 1px solid #eee;
        }}
        .result:last-child {{
            border-bottom: none;
        }}
        .prompt {{
            font-weight: 500;
            margin-bottom: 5px;
        }}
        .meta {{
            color: #666;
            font-size: 14px;
        }}
        .meta span {{
            margin-right: 15px;
        }}
        a {{
            color: #2563eb;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .top-prediction {{
            background: #e0f2fe;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <h1>Job Results: {summary['job_id']}</h1>

    <div class="summary">
        <div class="stat">
            <div class="stat-value">{summary['completed']}</div>
            <div class="stat-label">Completed</div>
        </div>
        <div class="stat">
            <div class="stat-value">{summary['failed']}</div>
            <div class="stat-label">Failed</div>
        </div>
        <div class="stat">
            <div class="stat-value">{summary['timing']['avg_duration_seconds']:.1f}s</div>
            <div class="stat-label">Avg Duration</div>
        </div>
        <div class="stat">
            <div class="stat-value">{summary['modules']['avg_modules_per_graph']:.1f}</div>
            <div class="stat-label">Avg Modules</div>
        </div>
    </div>

    <div class="results">
"""

    for i, output in enumerate(outputs):
        prompt = output.get("prompt", "Unknown")
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        n_modules = output.get("n_modules", 0)
        path = output.get("path", "")

        top_pred = ""
        top_logits = output.get("top_logits", [])
        if top_logits:
            top = top_logits[0]
            token = top.get("token", "")
            prob = top.get("probability", 0) * 100
            top_pred = f'<span class="top-prediction">{token} ({prob:.1f}%)</span>'

        html += f"""
        <div class="result">
            <div class="prompt">{prompt_preview}</div>
            <div class="meta">
                <span>Modules: {n_modules}</span>
                {top_pred}
                <span><a href="{path}">View JSON</a></span>
            </div>
        </div>
"""

    html += """
    </div>
</body>
</html>
"""

    output_path.write_text(html)


def copy_outputs_to_merged(manifest: JobManifest, merged_dir: Path) -> int:
    """Copy all output files to the merged directory."""
    count = 0
    for output_path in manifest.get_completed_outputs():
        src = Path(output_path)
        if src.exists():
            dst = merged_dir / src.name
            shutil.copy2(src, dst)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate batch job results"
    )
    parser.add_argument(
        "job_dir",
        type=Path,
        help="Path to job directory",
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="Generate HTML index page",
    )
    parser.add_argument(
        "--copy-outputs",
        action="store_true",
        help="Copy all output files to merged directory",
    )

    args = parser.parse_args()

    manifest = JobManifest(args.job_dir)
    if not manifest.exists():
        print(f"Error: No manifest found in {args.job_dir}", file=sys.stderr)
        sys.exit(1)

    # Generate summary
    print("Generating summary...")
    summary = generate_summary(manifest)

    # Create merged directory
    merged_dir = args.job_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary_path = merged_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_path}")

    # Print quick stats
    print(f"\nJob: {summary['job_id']}")
    print(f"  Completed: {summary['completed']}/{summary['total_prompts']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Avg duration: {summary['timing']['avg_duration_seconds']:.1f}s")
    print(f"  Total time: {summary['timing']['total_duration_seconds'] / 3600:.1f} hours")

    # Generate HTML index
    if args.create_index:
        index_path = merged_dir / "index.html"
        generate_html_index(summary, index_path)
        print(f"Created index: {index_path}")

    # Copy outputs
    if args.copy_outputs:
        count = copy_outputs_to_merged(manifest, merged_dir)
        print(f"Copied {count} output files to: {merged_dir}")


if __name__ == "__main__":
    main()
