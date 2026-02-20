"""Circuits - Attribution graph analysis for language models.

Main entry point:
    from circuits import run_pipeline

    result = run_pipeline("What is the capital of France?")

Individual components:
    - schemas: Common data types (Unit, Edge, Graph)
    - relp: RelP attribution computation
    - labeling: Neuron label fetching from database
    - clustering: Infomap-based neuron clustering
    - analysis: LLM-based circuit analysis
    - pipeline: Full pipeline orchestration
    - connectivity: Edge computation (output projections, weight graphs)
    - aggregation: Edge aggregation from many graphs
    - database: DuckDB circuit atlas builder/reader
    - autointerp: Progressive two-pass neuron labeling
    - cli: Unified CLI (``circuits graph``, ``circuits aggregate``, etc.)
"""

from .analysis import (
    analyze_modules,
    call_llm,
    compute_module_flow,
)
from .clustering import (
    cluster_full_model,
    cluster_graph,
    get_special_token_positions,
)
from .labeling import (
    fetch_descriptions,
    is_database_available,
    label_graph,
    label_graphs,
)
from .pipeline import (
    DEFAULT_OUTPUT_DIR,
    PipelineConfig,
    run_batch,
    run_from_config,
    run_pipeline,
    run_pipeline_from_graph,
)
from .relp import (
    RelPAttributor,
    RelPConfig,
    attribute,
    save_graph,
    validate_graph,
)
from .schemas import (
    ConnectivityMethod,
    Edge,
    Graph,
    LabelSource,
    TokenProjection,
    Unit,
    UnitLabel,
    UnitType,
)

__all__ = [
    # Schemas
    "Unit",
    "Edge",
    "Graph",
    "UnitType",
    "LabelSource",
    "ConnectivityMethod",
    "UnitLabel",
    "TokenProjection",
    # RelP attribution
    "RelPAttributor",
    "RelPConfig",
    "save_graph",
    "validate_graph",
    "attribute",
    # Labeling
    "label_graph",
    "label_graphs",
    "fetch_descriptions",
    "is_database_available",
    # Clustering
    "cluster_graph",
    "cluster_full_model",
    "get_special_token_positions",
    # Analysis
    "analyze_modules",
    "compute_module_flow",
    "call_llm",
    # Pipeline
    "run_pipeline",
    "run_pipeline_from_graph",
    "run_batch",
    "run_from_config",
    "PipelineConfig",
    "DEFAULT_OUTPUT_DIR",
]

