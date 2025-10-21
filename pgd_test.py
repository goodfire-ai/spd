# %%
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from spd.configs import PGDConfig
from spd.data import DatasetConfig, create_data_loader, loop_dataloader
from spd.eval import evaluate
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import replace_pydantic_model, set_seed

# %%

set_seed(0)

device = get_device()
run_info = SPDRunInfo.from_path("wandb:goodfire/spd/runs/3rthvni3")
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
batch_size = 64
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
data_iterator = loop_dataloader(data_loader)

pgd_config = [i for i in config.eval_metric_configs if i.classname == "PGDReconLoss"][0]
assert isinstance(pgd_config, PGDConfig)
pgd_config = replace_pydantic_model(pgd_config, {"mask_scope": mask_scope})

metrics = evaluate(
    eval_metric_configs=[pgd_config],
    model=model,
    eval_iterator=data_iterator,
    device=device,
    run_config=config,
    slow_step=True,
    n_eval_steps=1,
    current_frac_of_training=0.0,
)
print(f"Batch size: {batch_size}, Mask scope: {mask_scope}")
for k, v in metrics.items():
    print(f"{k}: {v}")

# Zero out all gradients
for param in model.parameters():
    param.grad = None


# %%