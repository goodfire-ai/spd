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
from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.models.component_model import ComponentModel, SPDRunInfo

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# magic autoreload
%load_ext autoreload
%autoreload 2

# %%
MODEL_PATH: str = "wandb:goodfire/spd/runs/ioprgffh"
SPD_RUN: SPDRunInfo = SPDRunInfo.from_path(MODEL_PATH)
component_model: ComponentModel = ComponentModel.from_pretrained(SPD_RUN.checkpoint_path)
component_model.to(DEVICE)
cfg: Config = SPD_RUN.config


# %%

N_SAMPLES: int = 1

dataset_config: DatasetConfig = DatasetConfig(
    name=cfg.task_config.dataset_name,
    hf_tokenizer_path=cfg.pretrained_model_name_hf,
    split=cfg.task_config.train_data_split,
    n_ctx=cfg.task_config.max_seq_len,
    is_tokenized=False,
    streaming=False,
	seed=0,
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

print(dataset[1])
print(next(iter(dataloader)))
print(dataset_config)