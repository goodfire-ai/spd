# %%
import time


class TimerContext:
    def __init__(self, message: str):
        self.message: str = message
        self.start_time: float

    def __enter__(self):
        print(f"[TIMER START] {self.message}")
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        elapsed_time: float = time.perf_counter() - self.start_time
        TIMER_RECORDS.append((self.message, elapsed_time))
        print(f"[TIMER   END] {self.message} - Elapsed time: {elapsed_time:.2f} seconds")


with TimerContext(message="Import modules"):
    from pathlib import Path

    import matplotlib.pyplot as plt
    import torch
    from jaxtyping import Int
    from muutils.dbg import dbg_auto
    from torch import Tensor

    from spd.clustering.activations import (
        ProcessedActivations,
        component_activations,
        process_activations,
    )
    from spd.clustering.merge import merge_iteration
    from spd.clustering.merge_config import MergeConfig
    from spd.clustering.merge_history import MergeHistory, MergeHistoryEnsemble
    from spd.clustering.merge_run_config import ClusteringRunConfig
    from spd.clustering.pipeline.s1_split_dataset import split_dataset
    from spd.clustering.plotting.activations import plot_activations
    from spd.clustering.plotting.merge import plot_dists_distribution
    from spd.models.component_model import ComponentModel, SPDRunInfo

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    TEMP_DIR: Path = Path(
        "tests/.temp"
    )  # save to an actual dir that is gitignored, so users can view plots
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    TIMER_RECORDS: list[tuple[str, float]] = list()


# magic autoreload
# %load_ext autoreload
# %autoreload 2

# %%
# Load model and dataset
# ============================================================
with TimerContext(message="Load model"):
    MODEL_PATH: str = "wandb:goodfire/spd-pre-Sep-2025/runs/ioprgffh"

    SPD_RUN: SPDRunInfo = SPDRunInfo.from_path(MODEL_PATH)
    MODEL: ComponentModel = ComponentModel.from_pretrained(SPD_RUN.checkpoint_path)
    MODEL.to(DEVICE)
    SPD_CONFIG = SPD_RUN.config

    # Use split_dataset with RunConfig to get real data
    CONFIG: ClusteringRunConfig = ClusteringRunConfig(
        merge_config=MergeConfig(),
        model_path=MODEL_PATH,
        task_name="lm",
        n_batches=1,
        batch_size=2,
    )

with TimerContext(message="Load data"):
    BATCHES, _ = split_dataset(
        config=CONFIG,
        config_kwargs=dict(streaming=True),
    )

# %%
# Load data batch
# ============================================================
with TimerContext(message="Load data batch"):
    DATA_BATCH: Int[Tensor, "batch_size n_ctx"] = next(BATCHES)

# %%
# Get component activations
# ============================================================
with TimerContext(message="Get component activations"):
    COMPONENT_ACTS: dict[str, Tensor] = component_activations(
        model=MODEL,
        batch=DATA_BATCH,
        device=DEVICE,
        sigmoid_type="hard",
    )

    _ = dbg_auto(COMPONENT_ACTS)
# %%
# Process activations
# ============================================================
with TimerContext(message="Process activations"):
    FILTER_DEAD_THRESHOLD: float = 0.001
    FILTER_MODULES: str = "model.layers.0"

    PROCESSED_ACTIVATIONS: ProcessedActivations = process_activations(
        activations=COMPONENT_ACTS,
        filter_dead_threshold=FILTER_DEAD_THRESHOLD,
        filter_modules=lambda x: x.startswith(FILTER_MODULES),
        seq_mode="concat",
    )

with TimerContext(message="Plot activations"):
    plot_activations(
        processed_activations=PROCESSED_ACTIVATIONS,
        save_dir=TEMP_DIR,
        n_samples_max=256,
        wandb_run=None,
        save_fmt="svg",
    )

# %%
# Compute ensemble merge iterations
# ============================================================
with TimerContext(message="Compute merge iterations"):
    MERGE_CFG: MergeConfig = MergeConfig(
        activation_threshold=0.01,
        alpha=0.01,
        iters=2,
        merge_pair_sampling_method="range",
        merge_pair_sampling_kwargs={"threshold": 0.1},
        pop_component_prob=0,
        module_name_filter=FILTER_MODULES,
        filter_dead_threshold=FILTER_DEAD_THRESHOLD,
    )

    # Modern approach: run merge_iteration multiple times to create ensemble
    ENSEMBLE_SIZE: int = 2
    HISTORIES: list[MergeHistory] = []
    for i in range(ENSEMBLE_SIZE):
        HISTORY: MergeHistory = merge_iteration(
            merge_config=MERGE_CFG,
            batch_id=f"batch_{i}",
            activations=PROCESSED_ACTIVATIONS.activations,
            component_labels=PROCESSED_ACTIVATIONS.labels,
            log_callback=None,
        )
        HISTORIES.append(HISTORY)

    ENSEMBLE: MergeHistoryEnsemble = MergeHistoryEnsemble(data=HISTORIES)


# %%
# Compute and plot distances
# ============================================================
with TimerContext(message="compute distances"):
    DISTANCES = ENSEMBLE.get_distances()


with TimerContext(message="plot distances"):
    plot_dists_distribution(
        distances=DISTANCES,
        mode="points",
    )
    plt.legend()


# %%
print("Timer records (s):")
for key, value in TIMER_RECORDS:
    print(f"{key:<30}{value:10.2f}")
