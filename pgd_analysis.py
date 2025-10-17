# %%
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# from spd.configs import PGDConfig
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import replace_pydantic_model, set_seed

# %%

set_seed(0)

device = get_device()
run_info = SPDRunInfo.from_path("wandb:goodfire/spd/runs/9d313yrl")
config = run_info.config
assert isinstance(config.task_config, LMTaskConfig), "task_config not LMTaskConfig"

# Change the task_config to use a sequence length of 10
task_config = replace_pydantic_model(
    config.task_config, {"max_seq_len": 512, "train_data_split": "train[:1000]"}
)

model = ComponentModel.from_run_info(run_info)
model.to(device)

# Set the target model params to frozen
model.target_model.requires_grad_(False)

train_data_config = DatasetConfig(
    name=task_config.dataset_name,
    hf_tokenizer_path=config.tokenizer_name,
    split=task_config.train_data_split,
    n_ctx=task_config.max_seq_len,
    is_tokenized=task_config.is_tokenized,
    streaming=task_config.streaming,
    column_name=task_config.column_name,
    shuffle_each_epoch=task_config.shuffle_each_epoch,
    seed=None,
)

# %%
# batch_sizes = [1, 8, 64]
# mask_scopes = ["unique_per_datapoint", "shared_across_batch"]
# from itertools import product

# for batch_size, mask_scope in product(batch_sizes, mask_scopes):
batch_size = 512
mask_scope = "shared_across_batch"
set_seed(0)
data_loader, _tokenizer = create_data_loader(
    dataset_config=train_data_config,
    batch_size=batch_size,
    buffer_size=task_config.buffer_size,
    global_seed=config.seed,
    ddp_rank=0,
    ddp_world_size=1,
)
# data_iterator = loop_dataloader(data_loader)
n_ones_list = []
# ITerate through the data iterator and check how many 1s there are in the input_ids tensor
for batch in data_loader:
    input_ids = batch["input_ids"]
    # Count the number of 1s in the input_ids tensor
    n_ones = (input_ids == 1).sum()
    n_ones_list.append(n_ones)

import numpy as np

print(f"Number of 1s: {n_ones_list}")
print(f"Mean number of 1s: {np.mean(n_ones_list)}")
print(f"Std number of 1s: {np.std(n_ones_list)}")
print(f"Min number of 1s: {np.min(n_ones_list)}")
print(f"Max number of 1s: {np.max(n_ones_list)}")
# %%
