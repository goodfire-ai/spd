# %%
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from spd.configs import PGDConfig
from spd.data import DatasetConfig, create_data_loader, loop_dataloader
from spd.eval import evaluate
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import set_seed

# %%

set_seed(0)

device = get_device()
run_info = SPDRunInfo.from_path("wandb:goodfire/spd/runs/opr3qkud")
config = run_info.config
assert isinstance(config.task_config, LMTaskConfig), "task_config not LMTaskConfig"

model = ComponentModel.from_run_info(run_info)
model.to(device)

# Set the target model params to frozen
model.target_model.requires_grad_(False)

train_data_config = DatasetConfig(
    name=config.task_config.dataset_name,
    hf_tokenizer_path=config.tokenizer_name,
    split=config.task_config.train_data_split,
    n_ctx=config.task_config.max_seq_len,
    is_tokenized=config.task_config.is_tokenized,
    streaming=config.task_config.streaming,
    column_name=config.task_config.column_name,
    shuffle_each_epoch=config.task_config.shuffle_each_epoch,
    seed=None,
)

# %%
set_seed(0)
batch_size = 1
data_loader, _tokenizer = create_data_loader(
    dataset_config=train_data_config,
    batch_size=batch_size,
    buffer_size=config.task_config.buffer_size,
    global_seed=config.seed,
    ddp_rank=0,
    ddp_world_size=1,
)
data_iterator = loop_dataloader(data_loader)

pgd_config = [i for i in config.eval_metric_configs if i.classname == "PGDReconLoss"][0]
assert isinstance(pgd_config, PGDConfig)

metrics = evaluate(
    metric_configs=[pgd_config],
    model=model,
    eval_iterator=data_iterator,
    device=device,
    run_config=config,
    slow_step=True,
    n_eval_steps=1,
    current_frac_of_training=0.0,
)
for k, v in metrics.items():
    print(f"{k}: {v}")

# Zero out all gradients
for param in model.parameters():
    param.grad = None


# %%
