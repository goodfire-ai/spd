"""End-to-end generic decomposition pipeline: harvest -> interpret -> evals."""

from dataclasses import dataclass
from pathlib import Path

from spd.autointerp_generic.db import GenericInterpDB
from spd.autointerp_generic.detection import DetectionResult, run_detection_scoring
from spd.autointerp_generic.fuzzing import FuzzingResult, run_fuzzing_scoring
from spd.autointerp_generic.interpret import run_interpret
from spd.autointerp_generic.intruder import IntruderResult, run_intruder_scoring
from spd.autointerp_generic.types import (
    ActivatingExample,
    ComponentAutointerpData,
    DecompositionAutointerpData,
    EvalConfig,
    InterpretationResult,
    InterpretConfig,
    IntruderConfig,
)
from spd.harvest.config import HarvestConfig
from spd.harvest.db import HarvestDB

from .harvest import harvest_decomposition
from .types import DecompositionSpec


@dataclass
class PipelineResults:
    harvest_db_path: Path
    interp_db_path: Path
    n_components: int
    interpretation_results: list[InterpretationResult]
    intruder_results: list[IntruderResult]
    detection_results: list[DetectionResult]
    fuzzing_results: list[FuzzingResult]


def _build_autointerp_data(
    spec: DecompositionSpec,
    harvest_db_path: Path,
) -> DecompositionAutointerpData:
    db = HarvestDB(harvest_db_path, readonly=True)
    components = db.get_all_components(activation_threshold=0.0)
    db.close()

    autointerp_components = [
        ComponentAutointerpData(
            key=component.component_key,
            component_explanation=spec.component_explanations.get(component.component_key, ""),
            activating_examples=[
                ActivatingExample(
                    tokens=example.token_ids,
                    bold=[v > spec.binarise_threshold for v in example.activation_values],
                )
                for example in component.activation_examples
            ],
        )
        for component in components
    ]

    return DecompositionAutointerpData(
        method=spec.method,
        decomposition_explanation=spec.decomposition_explanation,
        components=autointerp_components,
        tokenizer=spec.tokenizer,
    )


def run_pipeline(
    spec: DecompositionSpec,
    *,
    output_dir: Path,
    openrouter_api_key: str,
    harvest_config: HarvestConfig | None = None,
    interpret_config: InterpretConfig | None = None,
    intruder_config: IntruderConfig | None = None,
    eval_config: EvalConfig | None = None,
    n_gpus: int = 1,
    run_intruder: bool = True,
    run_detection: bool = True,
    run_fuzzing: bool = True,
    limit: int | None = None,
    cost_limit_usd: float | None = None,
) -> PipelineResults:
    """Run harvest -> interpret -> evals for any decomposition method."""
    if n_gpus != 1:
        raise NotImplementedError("run_pipeline currently supports n_gpus=1 only")

    harvest_config = harvest_config or HarvestConfig()
    interpret_config = interpret_config or InterpretConfig()
    intruder_config = intruder_config or IntruderConfig()
    eval_config = eval_config or EvalConfig()

    output_dir.mkdir(parents=True, exist_ok=True)
    harvest_decomposition(spec, harvest_config, output_dir)

    harvest_db_path = output_dir / "harvest.db"
    interp_db_path = output_dir / "interp.db"

    data = _build_autointerp_data(spec, harvest_db_path)
    interpretation_results = run_interpret(
        data=data,
        openrouter_api_key=openrouter_api_key,
        db_path=interp_db_path,
        config=interpret_config,
        limit=limit,
        cost_limit_usd=cost_limit_usd,
    )

    intruder_results: list[IntruderResult] = []
    detection_results: list[DetectionResult] = []
    fuzzing_results: list[FuzzingResult] = []

    if run_intruder:
        intruder_results = run_intruder_scoring(
            data=data,
            openrouter_api_key=openrouter_api_key,
            db_path=interp_db_path,
            config=intruder_config,
            limit=limit,
            cost_limit_usd=cost_limit_usd,
        )

    if run_detection or run_fuzzing:
        label_db = GenericInterpDB(interp_db_path, readonly=True)
        labels = label_db.get_labels()
        label_db.close()

        if run_detection:
            detection_results = run_detection_scoring(
                data=data,
                labels=labels,
                openrouter_api_key=openrouter_api_key,
                db_path=interp_db_path,
                config=eval_config,
                limit=limit,
                cost_limit_usd=cost_limit_usd,
            )

        if run_fuzzing:
            fuzzing_results = run_fuzzing_scoring(
                data=data,
                labels=labels,
                openrouter_api_key=openrouter_api_key,
                db_path=interp_db_path,
                config=eval_config,
                limit=limit,
                cost_limit_usd=cost_limit_usd,
            )

    return PipelineResults(
        harvest_db_path=harvest_db_path,
        interp_db_path=interp_db_path,
        n_components=len(data.components),
        interpretation_results=interpretation_results,
        intruder_results=intruder_results,
        detection_results=detection_results,
        fuzzing_results=fuzzing_results,
    )
