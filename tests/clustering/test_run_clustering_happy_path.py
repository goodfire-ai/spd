import tempfile
from pathlib import Path

import pytest

from spd.clustering.clustering_run_config import ClusteringRunConfig, LoggingIntervals
from spd.clustering.merge_config import MergeConfig
from spd.clustering.scripts.run_clustering import main


@pytest.mark.slow
def test_run_clustering_happy_path():
    """Test that run_clustering.py runs without errors."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ClusteringRunConfig(
            model_path="wandb:goodfire/spd/runs/zxbu57pt",  # An ss_llama run
            batch_size=4,
            dataset_seed=0,
            base_output_dir=Path(temp_dir),
            ensemble_id=None,
            merge_config=MergeConfig(
                activation_threshold=0.01,
                alpha=1.0,
                iters=3,
                merge_pair_sampling_method="range",
                merge_pair_sampling_kwargs={"threshold": 0.05},
            ),
            wandb_project=None,
            wandb_entity="goodfire",
            logging_intervals=LoggingIntervals(
                stat=1,
                tensor=100,
                plot=100,
                artifact=100,
            ),
            dataset_streaming=True,  # tests in CI very slow without this, see https://github.com/goodfire-ai/spd/pull/199
        )
        main(config)
