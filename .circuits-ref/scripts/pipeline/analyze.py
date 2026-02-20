#!/usr/bin/env python3
"""Run the complete attribution analysis pipeline.

This is the main entry point for analyzing prompts.

Usage:
    # Full pipeline from prompt
    python scripts/analyze.py "What is the capital of France?"

    # With answer prefix
    python scripts/analyze.py "The capital of France is" --answer-prefix " Paris"

    # From existing graph file
    python scripts/analyze.py --graph graphs/my-graph.json

    # From config file (batch mode)
    python scripts/analyze.py --config config.yaml

    # Generate example config
    python scripts/analyze.py --generate-config

    # With functional splitting
    python scripts/analyze.py "Your prompt" --functional-split

    # Show current config
    python scripts/analyze.py --show-config

Examples:
    # Medical knowledge circuit
    python scripts/analyze.py "Parkinson's disease involves degeneration of neurons in the"

    # Factual recall
    python scripts/analyze.py "The Eiffel Tower is located in" --answer-prefix " Paris"

    # Batch from config
    python scripts/analyze.py --config configs/medical_prompts.yaml
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from circuits.pipeline import (
    DEFAULT_OUTPUT_DIR,
    PipelineConfig,
    run_from_config,
    run_pipeline,
    run_pipeline_from_graph,
)


def main():
    parser = argparse.ArgumentParser(
        description='Run attribution analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('prompt', nargs='?', help='Prompt to analyze')
    input_group.add_argument('--graph', type=Path, help='Existing graph file to analyze')
    input_group.add_argument('--config', type=Path, help='YAML config file for batch processing')
    input_group.add_argument('--generate-config', action='store_true',
                             help='Generate example config file')
    input_group.add_argument('--show-config', action='store_true',
                             help='Show default configuration')

    # Model options
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct',
                        help='Model name or path (default: meta-llama/Llama-3.1-8B-Instruct)')

    # Template options
    parser.add_argument('--answer-prefix', default='',
                        help='Prefill assistant response (e.g., " Answer:")')
    parser.add_argument('--raw', action='store_true',
                        help='Use raw prompt without chat template')

    # Pipeline options
    parser.add_argument('--no-labels', action='store_true',
                        help='Skip neuron labeling from database')
    parser.add_argument('--no-llm', action='store_true',
                        help='Skip LLM analysis (just cluster)')
    parser.add_argument('--llm-model', default='auto',
                        help='LLM model for analysis (default: auto)')
    parser.add_argument('--llm-provider', default='auto',
                        choices=['auto', 'openai', 'anthropic'],
                        help='LLM provider (default: auto)')

    # Clustering options
    parser.add_argument('--min-cluster', type=int, default=5,
                        help='Min cluster size to subdivide (default: 5)')
    parser.add_argument('--max-depth', type=int, default=1,
                        help='Max clustering recursion depth (default: 1 = subdivide once)')
    parser.add_argument('--include-special-tokens', action='store_true',
                        help='Include special tokens (BOS, headers) in clustering')

    # Functional splitting options (enabled by default with LLM split)
    parser.add_argument('--no-functional-split', action='store_true',
                        help='Disable functional sub-module splitting (enabled by default)')
    parser.add_argument('--functional-split-min-size', type=int, default=2,
                        help='Only split modules with at least this many neurons (default: 2)')
    parser.add_argument('--no-prompt-answer-split', action='store_true',
                        help='Disable prompt vs answer splitting (first split)')
    parser.add_argument('--no-position-split', action='store_true',
                        help='Disable position-based splitting (contiguous token spans)')
    parser.add_argument('--max-position-gap', type=int, default=3,
                        help='Max gap between token positions before splitting (default: 3)')
    parser.add_argument('--layer-split', action='store_true',
                        help='Enable layer-based splitting (early/mid/late) - disabled by default')
    parser.add_argument('--semantic-split', action='store_true',
                        help='Enable semantic splitting (neuron label clustering) - disabled by default, use llm-split instead')
    parser.add_argument('--no-llm-split', action='store_true',
                        help='Disable LLM-based functional splitting (enabled by default)')
    parser.add_argument('--no-llm-reassignment', action='store_true',
                        help='Disable LLM module reassignment step (enabled by default)')

    # Graph generation options
    parser.add_argument('--k', type=int, default=5,
                        help='Number of top logits (default: 5)')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Node threshold (default: 0.005)')
    parser.add_argument('--target-tokens', type=str, nargs='+',
                        help='Specific tokens to trace (e.g., " Yes" " No"). Overrides --k')
    parser.add_argument('--contrastive', type=str, nargs=2, metavar=('POS', 'NEG'),
                        help='Contrastive attribution: trace logit(POS) - logit(NEG). E.g., --contrastive Yes No')

    # Output options
    parser.add_argument('--output', '-o', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--no-intermediate', action='store_true',
                        help='Don\'t save intermediate files')

    # Other
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress messages')

    args = parser.parse_args()

    # Handle special commands
    if args.generate_config:
        config_path = Path("config.yaml")
        PipelineConfig.generate_example(config_path)
        print(f"Generated example config: {config_path}")
        return

    if args.show_config:
        config = PipelineConfig()
        print(config)
        return

    # Validate that we have some input
    if not args.prompt and not args.graph and not args.config:
        parser.print_help()
        print("\nError: Must provide a prompt, --graph, or --config")
        sys.exit(1)

    verbose = not args.quiet

    # Run from config file (batch mode)
    if args.config:
        results = run_from_config(args.config, verbose=verbose)

        print(f"\n{'='*60}")
        print(f"Batch complete: {len(results)} sequences")
        print(f"{'='*60}")
        for r in results:
            print(f"  {r['output_path']}")

        print("\nTo view the analyses:")
        print("  1. Open frontend/flow_viewer.html in a browser")
        print("  2. Click 'Load JSON' and select any output file")
        return

    # Build config from CLI args
    config = PipelineConfig(
        model_name=args.model,
        use_chat_template=not args.raw,
        answer_prefix=args.answer_prefix,
        label_neurons=not args.no_labels,
        run_llm_analysis=not args.no_llm,
        llm_model=args.llm_model,
        llm_provider=args.llm_provider,
        skip_special_tokens=not args.include_special_tokens,
        min_cluster_size=args.min_cluster,
        max_cluster_depth=args.max_depth,
        k=args.k,
        tau=args.tau,
        target_tokens=args.target_tokens,
        contrastive_tokens=args.contrastive,
        output_dir=args.output,
        save_intermediate=not args.no_intermediate,
        # Functional splitting (enabled by default)
        functional_split=not args.no_functional_split,
        functional_split_min_size=args.functional_split_min_size,
        use_prompt_answer_split=not args.no_prompt_answer_split,
        use_position_split=not args.no_position_split,
        max_position_gap=args.max_position_gap,
        use_layer_split=args.layer_split,
        use_semantic_split=args.semantic_split,
        use_llm_split=not args.no_llm_split,
        use_llm_reassignment=not args.no_llm_reassignment,
    )

    # Run pipeline
    if args.graph:
        # From existing graph
        result = run_pipeline_from_graph(
            args.graph,
            config=config,
            verbose=verbose
        )
    else:
        # From prompt
        result = run_pipeline(
            args.prompt,
            config=config,
            verbose=verbose
        )

    # Print final output location
    print(f"\nOutput: {result['output_path']}")

    # Print viewing instructions
    print("\nTo view the analysis:")
    print("  1. Open frontend/flow_viewer.html in a browser")
    print(f"  2. Click 'Load JSON' and select: {result['output_path']}")


if __name__ == '__main__':
    main()
