"""
Refactored clustering module with clean architecture.

Key improvements:
- Replaced 370+ lines of subprocess/FD code with simple multiprocessing.Pool
- Unified functions with optional callbacks instead of separate versions
- Clean 3-layer separation: computation, data processing, orchestration
"""

# from spd.clustering.refactored.data_processing import (
#     extract_and_process_activations,
#     load_batch_data,
#     load_merge_histories,
#     load_merge_history,
#     process_single_batch,
#     save_distances,
#     save_merge_history,
#     save_normalized_data,
# )
# from spd.clustering.refactored.merge import merge_iteration
# from spd.clustering.refactored.orchestration import (
#     ClusteringResults,
#     cluster_analysis_pipeline,
#     process_batches_parallel,
# )
# from spd.clustering.refactored.wandb_setup import create_wandb_setup

# __all__ = [
#     # Data processing
#     "extract_and_process_activations",
#     "load_batch_data",
#     "load_merge_histories",
#     "load_merge_history",
#     "process_single_batch",
#     "save_distances",
#     "save_merge_history",
#     "save_normalized_data",
#     # Core algorithm
#     "merge_iteration",
#     # Orchestration
#     "ClusteringResults",
#     "cluster_analysis_pipeline",
#     "process_batches_parallel",
#     # WandB
#     "create_wandb_setup",
# ]