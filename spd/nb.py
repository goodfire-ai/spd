# ruff: noqa

# %%
from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader, loop_dataloader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import extract_batch_data, load_config, resolve_class

# %%

path = "spd/experiments/lm/ss_llama_config.yaml"

config = load_config(path, config_model=Config)

device = get_device()
assert isinstance(config.task_config, LMTaskConfig), "task_config not LMTaskConfig"

pretrained_model_class = resolve_class(config.pretrained_model_class)
assert hasattr(pretrained_model_class, "from_pretrained"), (
    f"Model class {pretrained_model_class} should have a `from_pretrained` method"
)
assert config.pretrained_model_name is not None
# %%
target_model = pretrained_model_class.from_pretrained(config.pretrained_model_name)  # pyright: ignore[reportAttributeAccessIssue]

target_model.eval()
# %%

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

train_loader, _tokenizer = create_data_loader(
    dataset_config=train_data_config,
    batch_size=config.batch_size,
    buffer_size=config.task_config.buffer_size,
    global_seed=config.seed,
    ddp_rank=0,
    ddp_world_size=1,
)

eval_data_config = DatasetConfig(
    name=config.task_config.dataset_name,
    hf_tokenizer_path=config.tokenizer_name,
    split=config.task_config.eval_data_split,
    n_ctx=config.task_config.max_seq_len,
    is_tokenized=config.task_config.is_tokenized,
    streaming=config.task_config.streaming,
    column_name=config.task_config.column_name,
    shuffle_each_epoch=config.task_config.shuffle_each_epoch,
    seed=None,
)

# %%

# %%

eval_loader, _ = create_data_loader(
    dataset_config=eval_data_config,
    batch_size=config.eval_batch_size,
    buffer_size=config.task_config.buffer_size,
    global_seed=config.seed + 1,
    ddp_rank=0,
    ddp_world_size=1,
)

# %%

target_model.requires_grad_(False)

model = ComponentModel(
    target_model=target_model,
    target_module_patterns=config.all_module_patterns,
    C=config.C,
    ci_fn_type=config.ci_fn_type,
    ci_fn_hidden_dims=config.ci_fn_hidden_dims,
    pretrained_model_output_attr=config.pretrained_model_output_attr,
)
model.to(device)

# %%

dl_iter = loop_dataloader(train_loader)


def b():
    return extract_batch_data(next(dl_iter)).to(device)


# %%

b().shape
# # %%
