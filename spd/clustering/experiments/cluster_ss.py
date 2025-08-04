# %%

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from muutils.dbg import dbg_auto

from spd.clustering.activations import component_activations, process_activations
from spd.clustering.merge import (
    MergeConfig,
    MergeEnsemble,
    merge_iteration_ensemble,
)
from spd.clustering.plotting.merge import plot_dists_distribution
from spd.data import DatasetConfig, create_data_loader
from spd.models.component_model import ComponentModel, SPDRunInfo

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# magic autoreload
# %load_ext autoreload
# %autoreload 2

# %%
SPD_RUN = SPDRunInfo.from_path("wandb:goodfire/spd/runs/ioprgffh")
component_model: ComponentModel = ComponentModel.from_pretrained(SPD_RUN.checkpoint_path)
component_model.to(DEVICE)
cfg = SPD_RUN.config


# %%
dbg_auto(component_model.state_dict())
dbg_auto(cfg)
dbg_auto(cfg.task_config)
# %%

N_SAMPLES: int = 4

dataset_config = DatasetConfig(
    name=cfg.task_config.dataset_name,
    hf_tokenizer_path=cfg.pretrained_model_name_hf,
    split=cfg.task_config.train_data_split,
    n_ctx=cfg.task_config.max_seq_len,
    is_tokenized=False,
    streaming=False,
    column_name=cfg.task_config.column_name,
)

dataset = load_dataset(
    dataset_config.name,
    streaming=dataset_config.streaming,
    split=dataset_config.split,
    trust_remote_code=False,
)

dataloader, _tokenizer = create_data_loader(
    dataset_config=dataset_config,
    batch_size=N_SAMPLES,
    buffer_size=cfg.task_config.buffer_size,
    global_seed=cfg.seed,
    ddp_rank=0,
    ddp_world_size=1,
)


# %%

ci = component_activations(
    component_model,
    dataloader,
    device=DEVICE,
    # threshold=0.1,
    # TODO: where can we find this in the model itself???
    sigmoid_type="hard",
)

dbg_auto(ci)
# %%
coa = process_activations(
    ci,
    filter_dead_threshold=0.001,
    seq_mode="concat",
    plots=True,  # Plot the processed activations
    # plot_title="Processed Activations",
)
# %%


ENSEMBLE: MergeEnsemble = merge_iteration_ensemble(
    activations=coa["activations"],
    component_labels=coa["labels"],
    merge_config=MergeConfig(
        activation_threshold=None,
        alpha=0.01,
        iters=100,
        check_threshold=0.1,
        pop_component_prob=0,
        rank_cost_fn=lambda x: 1.0,
    ),
    ensemble_size=16,
)


# %%
DISTANCES = ENSEMBLE.get_distances()


# %%
plot_dists_distribution(
    distances=DISTANCES,
    mode="points",
    # label="v1"
)
plt.legend()
