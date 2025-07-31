# %%

import matplotlib.pyplot as plt
import torch
from muutils.dbg import dbg_auto

from spd.clustering.activations import component_activations, process_activations
from spd.clustering.merge import (
    MergeConfig,
    MergePlotConfig,
    merge_iteration_ensemble,
	MergeEnsemble,
)
from spd.clustering.plotting.merge import plot_dists_distribution
from spd.data import DatasetConfig, create_data_loader
from datasets import load_dataset
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.models.component_model import ComponentModel
from spd.registry import CANONICAL_RUNS
from spd.utils.data_utils import DatasetGeneratedDataLoader

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# magic autoreload
%load_ext autoreload
%autoreload 2

# %%
component_model, cfg, path = ComponentModel.from_pretrained("wandb:goodfire/spd/runs/ioprgffh")
# component_model, cfg, path = ComponentModel.from_pretrained(CANONICAL_RUNS["tms_40-10-id"])
component_model.to(DEVICE);

# %%
dbg_auto(component_model.state_dict()); 

dbg_auto(cfg);
dbg_auto(cfg.task_config);

# %%

N_SAMPLES: int = 8

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

dbg_auto(ci);
# %%
coa = process_activations(
    ci,
    filter_dead_threshold=0.001,
	seq_mode="seq_mean",
    plots=True,  # Plot the processed activations
    # plot_title="Processed Activations",
);

# %%


ENSEMBLE: MergeEnsemble = merge_iteration_ensemble(
    activations=coa["activations"],
    component_labels=coa["labels"],
    merge_config=MergeConfig(
        activation_threshold=None,
        alpha=0.01,
        iters=10,
        check_threshold=0.1,
        pop_component_prob=0.1,
        rank_cost_fn=lambda x: 1.0,
        stopping_condition=None,
    ),
	ensemble_size=16,
)
# %%
DISTANCES = ENSEMBLE.get_distances()


# %%
plot_dists_distribution(
	distances=DISTANCES,
	mode="dist",
	# label="v1"
)
plt.legend()
