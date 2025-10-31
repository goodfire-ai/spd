# %%
import warnings

from spd.metrics.pgd_utils import calc_pgd_global_masked_recon_loss

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from spd.configs import PGDGlobalReconLossConfig
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import replace_pydantic_model, set_seed

# %%


device = get_device()
run_info = SPDRunInfo.from_path("wandb:goodfire/spd/runs/lxs77xye")
config = run_info.config
assert isinstance(config.task_config, LMTaskConfig), "task_config not LMTaskConfig"

# %%

batch_size = 2
# max_seq_len = 512
max_seq_len = 8
n_batches = 20

for n_batches in [1, 2, 4, 8, 16, 32, 64]:
    task_config = replace_pydantic_model(
        config.task_config, {"max_seq_len": max_seq_len, "train_data_split": "train"}
    )

    model = ComponentModel.from_run_info(run_info)
    model.to(device)
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

    pgd_global_config = PGDGlobalReconLossConfig(
        init="random",
        step_size=0.1,
        n_steps=20,
        n_batches=n_batches,
    )

    set_seed(0)
    data_loader, _tokenizer = create_data_loader(
        dataset_config=train_data_config,
        batch_size=batch_size,
        buffer_size=task_config.buffer_size,
        global_seed=config.seed,
        ddp_rank=0,
        ddp_world_size=1,
    )

    loss = calc_pgd_global_masked_recon_loss(
        pgd_config=pgd_global_config,
        model=model,
        dataloader=data_loader,
        output_loss_type=config.output_loss_type,
        routing="all",
        sampling=config.sampling,
        use_delta_component=config.use_delta_component,
        batch_dims=(batch_size, task_config.max_seq_len),
    )
    print(f"n_batches: {n_batches}, batch_size: {batch_size}, seq_len: {max_seq_len}, Loss: {loss}")


# %%
