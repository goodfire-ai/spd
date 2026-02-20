"""Unified CLI for the circuits pipeline.

Usage:
    circuits graph "prompt"              # One-off RelP graph
    circuits analyze --config x.yaml     # Full single-prompt analysis
    circuits aggregate <graph_dir>       # Aggregate edges from many graphs
    circuits label --config x.yaml       # Two-pass autointerp sweep
    circuits cluster --duckdb x.db       # Cluster full model
    circuits build-db --config x.yaml    # Build DuckDB atlas
    circuits query <db> "pattern"        # Search atlas

Can also be invoked as:
    python -m circuits.cli <subcommand> [args]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_graph(args: argparse.Namespace) -> None:
    """Generate a RelP attribution graph from a prompt."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .pipeline import apply_chat_template, format_for_viewer, slugify
    from .relp import RelPAttributor, RelPConfig

    prompt = args.prompt
    model_name = args.model
    slug = args.slug or f"relp-{slugify(prompt)}"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_name}")
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Format prompt
    if args.raw:
        formatted = prompt
    else:
        formatted = apply_chat_template(tokenizer, prompt, args.answer_prefix or "")

    # Build RelP config
    target_tokens = args.target_tokens
    relp_config = RelPConfig(
        k=args.k,
        tau=args.tau,
        target_tokens=target_tokens,
        use_neuron_labels=False,
    )

    print(f"Generating graph for: {prompt[:60]}...")
    attributor = RelPAttributor(model, tokenizer, config=relp_config)
    graph = attributor.compute_attributions(formatted, target_tokens=target_tokens)
    attributor.cleanup()

    graph["metadata"]["prompt"] = prompt
    graph = format_for_viewer(graph, slug)

    graph_path = output_dir / f"{slug}.json"
    with open(graph_path, "w") as f:
        json.dump(graph, f, indent=2)

    print(f"Saved graph ({len(graph['nodes'])} nodes, {len(graph['links'])} edges) to {graph_path}")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run the full analysis pipeline."""
    from .pipeline import PipelineConfig, run_from_config, run_pipeline

    if args.config:
        results = run_from_config(args.config, verbose=not args.quiet)
        print(f"Analyzed {len(results)} sequences")
    elif args.prompt:
        config = PipelineConfig()
        if args.output:
            config.output_dir = Path(args.output)
        result = run_pipeline(args.prompt, config=config, verbose=not args.quiet)
        print(f"Output: {result['output_path']}")
    else:
        print("Error: provide --config or a prompt string", file=sys.stderr)
        sys.exit(1)


def cmd_aggregate(args: argparse.Namespace) -> None:
    """Aggregate edges from many RelP graphs."""
    from .aggregation import InMemoryAggregator

    graph_dir = Path(args.graph_dir)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else graph_dir / "checkpoints"

    agg = InMemoryAggregator(graph_dir=graph_dir, checkpoint_dir=checkpoint_dir)

    # Resume from checkpoint if available
    latest = checkpoint_dir / "latest.dat"
    if latest.exists() and not args.fresh:
        print(f"Resuming from checkpoint: {latest}")
        agg.resume(latest)
        status = agg.get_status()
        print(f"  Graphs processed: {status['graphs_processed']}")
        print(f"  Unique edges: {status['edges_count']}")

    # Process directory
    agg.process_directory(
        graph_dir,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        max_graphs=args.max_graphs,
    )

    status = agg.get_status()
    print(f"\nFinal: {status['graphs_processed']} graphs, {status['edges_count']} edges")

    # Export if requested
    if args.export_duckdb:
        print(f"Exporting to DuckDB: {args.export_duckdb}")
        agg.export_to_duckdb(Path(args.export_duckdb))
        print("Done.")


def cmd_label(args: argparse.Namespace) -> None:
    """Run two-pass autointerp labeling."""
    from .autointerp import ProgressiveInterpreter

    interp = ProgressiveInterpreter(
        model_name=args.model,
        edge_stats_path=args.edge_stats,
        db_path=args.db_path or "data/neuron_functions.json",
    )

    if args.load_model:
        interp.load_model()
    interp.load_edge_stats()

    start = args.start_layer
    end = args.end_layer
    llm_model = args.llm_model or "gpt-5.2-mini"

    print(f"Running two-pass labeling: layers {start}→{end}, LLM={llm_model}")
    interp.llm_label_all_layers(
        start_layer=start,
        end_layer=end,
        model=llm_model,
        passes=args.passes,
    )
    print("Done.")


def cmd_cluster(args: argparse.Namespace) -> None:
    """Cluster neurons (full model or single graph)."""
    if args.graph:
        # Single-graph clustering
        from .clustering import cluster_graph

        with open(args.graph) as f:
            graph_data = json.load(f)

        result = cluster_graph(graph_data, verbose=True)
        print(f"Found {result['methods'][0]['n_clusters']} clusters")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved to {args.output}")
    elif args.duckdb:
        # Full-model clustering from DuckDB
        from .clustering import cluster_full_model
        from .database import CircuitDatabase

        with CircuitDatabase(args.duckdb, read_only=True) as db:
            print("Loading units and edges from database...")
            # Get all edges
            edges_rows = db._conn.execute(
                "SELECT * FROM edges WHERE count >= ?",
                (args.min_edge_count,),
            ).fetchall()
            edges = [db._row_to_edge(r) for r in edges_rows]

            # Get all units
            unit_rows = db._conn.execute("SELECT * FROM neurons").fetchall()
            units = [db._row_to_unit(r) for r in unit_rows]

        print(f"Loaded {len(units)} units, {len(edges)} edges")
        result = cluster_full_model(
            edges, units,
            min_edge_count=args.min_edge_count,
            weight_transform=args.weight_transform,
            verbose=True,
        )

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved {len(result['assignments'])} assignments to {args.output}")
    else:
        print("Error: provide --graph or --duckdb", file=sys.stderr)
        sys.exit(1)


def cmd_build_db(args: argparse.Namespace) -> None:
    """Build DuckDB atlas from aggregated data."""
    from .aggregation import InMemoryAggregator
    from .database import CircuitDatabase

    graph_dir = Path(args.graph_dir)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else graph_dir / "checkpoints"
    output_path = Path(args.output)

    # Load aggregator from checkpoint
    agg = InMemoryAggregator(graph_dir=graph_dir, checkpoint_dir=checkpoint_dir)
    latest = checkpoint_dir / "latest.dat"
    if not latest.exists():
        print(f"Error: no checkpoint found at {latest}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint: {latest}")
    agg.resume(latest)
    status = agg.get_status()
    print(f"  Graphs: {status['graphs_processed']}, Edges: {status['edges_count']}")

    # Load cluster assignments if provided
    cluster_assignments = None
    if args.clusters:
        print(f"Loading cluster assignments: {args.clusters}")
        with open(args.clusters) as f:
            cluster_assignments = json.load(f)
        print(f"  {len(cluster_assignments)} assignments")

    # Build database
    print(f"Building DuckDB: {output_path}")
    db = CircuitDatabase.build_from_aggregator(
        agg,
        labels_path=Path(args.labels) if args.labels else None,
        cluster_assignments=cluster_assignments,
        output_path=output_path,
        min_edge_count=args.min_edge_count,
    )
    stats = db.get_stats()
    db.close()
    print(f"Done: {stats['neurons_count']} neurons, {stats['edges_count']} edges")


def cmd_query(args: argparse.Namespace) -> None:
    """Search the atlas database."""
    from .database import CircuitDatabase

    with CircuitDatabase(args.db, read_only=True) as db:
        if args.sql:
            # Raw SQL query
            result = db._conn.execute(args.pattern).fetchall()
            for row in result[:args.limit]:
                print(row)
        else:
            # Label search
            units = db.search_units(f"%{args.pattern}%", limit=args.limit)
            if not units:
                print("No matches found.")
                return
            for u in units:
                cluster = f" [cluster {u.top_cluster}]" if u.top_cluster is not None else ""
                print(f"  {u.unit_id}: {u.label}{cluster}")
            print(f"\n{len(units)} results")


# ---------------------------------------------------------------------------
# Argument parsers
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="circuits",
        description="Unified CLI for the circuits pipeline",
    )
    sub = parser.add_subparsers(dest="command", help="Subcommands")

    # -- graph --
    p_graph = sub.add_parser("graph", help="Generate a RelP attribution graph")
    p_graph.add_argument("prompt", help="Input prompt")
    p_graph.add_argument("--slug", help="Output filename slug")
    p_graph.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p_graph.add_argument("--device", default="cuda")
    p_graph.add_argument("--dtype", default="bfloat16")
    p_graph.add_argument("--k", type=int, default=5, help="Top logits to trace")
    p_graph.add_argument("--tau", type=float, default=0.005, help="Node threshold")
    p_graph.add_argument("--target-tokens", nargs="+", help="Specific tokens to trace")
    p_graph.add_argument("--raw", action="store_true", help="Skip chat template")
    p_graph.add_argument("--answer-prefix", default="", help="Prefill assistant response")
    p_graph.add_argument("-o", "--output", default="graphs/", help="Output directory")

    # -- analyze --
    p_analyze = sub.add_parser("analyze", help="Full analysis pipeline")
    p_analyze.add_argument("prompt", nargs="?", help="Input prompt")
    p_analyze.add_argument("--config", help="YAML config file")
    p_analyze.add_argument("-o", "--output", help="Output directory")
    p_analyze.add_argument("-q", "--quiet", action="store_true")

    # -- aggregate --
    p_agg = sub.add_parser("aggregate", help="Aggregate edges from many graphs")
    p_agg.add_argument("graph_dir", help="Directory containing graph JSONs")
    p_agg.add_argument("--checkpoint-dir", help="Checkpoint directory")
    p_agg.add_argument("--batch-size", type=int, default=100)
    p_agg.add_argument("--checkpoint-interval", type=int, default=500)
    p_agg.add_argument("--max-graphs", type=int, default=0, help="0 = unlimited")
    p_agg.add_argument("--export-duckdb", help="Export to DuckDB after aggregation")
    p_agg.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints")

    # -- label --
    p_label = sub.add_parser("label", help="Two-pass autointerp labeling")
    p_label.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p_label.add_argument("--edge-stats", required=True, help="Edge stats JSON path")
    p_label.add_argument("--db-path", help="Neuron function DB path")
    p_label.add_argument("--llm-model", default="gpt-5.2-mini")
    p_label.add_argument("--start-layer", type=int, default=31)
    p_label.add_argument("--end-layer", type=int, default=0)
    p_label.add_argument("--passes", type=int, default=2)
    p_label.add_argument("--load-model", action="store_true", help="Load the neural network model (for projections)")

    # -- cluster --
    p_cluster = sub.add_parser("cluster", help="Cluster neurons")
    p_cluster.add_argument("--graph", help="Single graph JSON for per-graph clustering")
    p_cluster.add_argument("--duckdb", help="DuckDB atlas for full-model clustering")
    p_cluster.add_argument("--min-edge-count", type=int, default=5)
    p_cluster.add_argument("--weight-transform", default="abs_weight_sq",
                           choices=["abs_weight", "abs_weight_sq", "weight"])
    p_cluster.add_argument("-o", "--output", help="Output JSON path")

    # -- build-db --
    p_db = sub.add_parser("build-db", help="Build DuckDB atlas")
    p_db.add_argument("graph_dir", help="Directory containing graph JSONs")
    p_db.add_argument("--checkpoint-dir", help="Checkpoint directory")
    p_db.add_argument("--labels", help="JSONL labels file")
    p_db.add_argument("--clusters", help="Cluster assignments JSON")
    p_db.add_argument("--min-edge-count", type=int, default=3)
    p_db.add_argument("-o", "--output", default="data/atlas.duckdb", help="Output DuckDB path")

    # -- query --
    p_query = sub.add_parser("query", help="Search the atlas database")
    p_query.add_argument("db", help="DuckDB database path")
    p_query.add_argument("pattern", help="Search pattern (label substring or SQL)")
    p_query.add_argument("--sql", action="store_true", help="Treat pattern as raw SQL")
    p_query.add_argument("--limit", type=int, default=20)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "graph": cmd_graph,
        "analyze": cmd_analyze,
        "aggregate": cmd_aggregate,
        "label": cmd_label,
        "cluster": cmd_cluster,
        "build-db": cmd_build_db,
        "query": cmd_query,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
