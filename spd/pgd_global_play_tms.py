# %%
import warnings

from spd.metrics.pgd_utils import calc_pgd_global_masked_recon_loss

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from spd.configs import PGDGlobalReconLossConfig
from spd.experiments.tms.configs import TMSTaskConfig
from spd.experiments.tms.models import TMSModel, TMSTargetRunInfo
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import set_seed

# %%
device = get_device()
# run_info = SPDRunInfo.from_path("wandb:goodfire/spd/runs/amyo423r")  # tms_40-10
run_info = SPDRunInfo.from_path("wandb:goodfire/spd/runs/kyywp29j")  # tms_40-10-id
config = run_info.config
task_config = config.task_config
assert isinstance(task_config, TMSTaskConfig), "task_config not TMSTaskConfig"

# %%

n_batches = 1
# for n_batches in [1, 2, 4, 8, 16, 32, 64]:
# for batch_size in [1, 2, 4, 8, 16, 32, 64]:
for batch_size in [1, 2, 4, 8, 16, 32, 64]:
    model = ComponentModel.from_run_info(run_info)
    model.to(device)
    target_model = model.target_model
    assert isinstance(target_model, TMSModel), "target_model not TMSModel"
    target_model.requires_grad_(False)

    assert config.pretrained_model_path, "pretrained_model_path must be set"
    target_run_info = TMSTargetRunInfo.from_path(config.pretrained_model_path)

    set_seed(0)
    synced_inputs = target_run_info.config.synced_inputs
    dataset = SparseFeatureDataset(
        n_features=target_model.config.n_features,
        feature_probability=task_config.feature_probability,
        device=device,
        data_generation_type=task_config.data_generation_type,
        value_range=(0.0, 1.0),
        synced_inputs=synced_inputs,
    )
    data_loader = DatasetGeneratedDataLoader(dataset, batch_size=batch_size, shuffle=False)

    pgd_global_config = PGDGlobalReconLossConfig(
        init="random",
        step_size=0.1,
        n_steps=20,
        gradient_accumulation_steps=n_batches,
    )

    loss = calc_pgd_global_masked_recon_loss(
        pgd_config=pgd_global_config,
        model=model,
        dataloader=data_loader,
        output_loss_type=config.output_loss_type,
        routing="all",
        sampling=config.sampling,
        use_delta_component=config.use_delta_component,
        batch_dims=(batch_size,),
    )
    print(f"n_batches: {n_batches}, batch_size: {batch_size}, Loss: {loss}")


# %%
