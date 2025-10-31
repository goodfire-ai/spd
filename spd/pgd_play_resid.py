from spd.configs import PGDReconLossConfig
from spd.experiments.resid_mlp.configs import ResidMLPTaskConfig
from spd.experiments.resid_mlp.models import ResidMLP, ResidMLPTargetRunInfo
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
from spd.metrics.pgd_masked_recon_loss import PGDReconLoss
from spd.models.component_model import ComponentModel, OutputWithCache, SPDRunInfo
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import extract_batch_data, set_seed

device = get_device()
run_info = SPDRunInfo.from_path("wandb:goodfire/spd/runs/grevt2h2")  # resid_mlp2
# run_info = SPDRunInfo.from_path("wandb:goodfire/spd/runs/uz5swum7")  # resid_mlp1
config = run_info.config
task_config = config.task_config
assert isinstance(task_config, ResidMLPTaskConfig), "task_config not ResidMLPTaskConfig"


model = ComponentModel.from_run_info(run_info)
model.to(device)
target_model = model.target_model
assert isinstance(target_model, ResidMLP), "target_model not ResidMLP"
target_model.requires_grad_(False)

assert config.pretrained_model_path, "pretrained_model_path must be set"
target_run_info = ResidMLPTargetRunInfo.from_path(config.pretrained_model_path)

set_seed(0)
synced_inputs = target_run_info.config.synced_inputs
dataset = ResidMLPDataset(
    n_features=target_model.config.n_features,
    feature_probability=task_config.feature_probability,
    device=device,
    calc_labels=False,  # Our labels will be the output of the target model
    label_type=None,
    act_fn_name=None,
    label_fn_seed=None,
    label_coeffs=None,
    data_generation_type=task_config.data_generation_type,
    synced_inputs=synced_inputs,
)

n_batches = 1
seed = 0
mask_scope = "unique_per_datapoint"
# mask_scope = "shared_across_batch"
for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    set_seed(seed)
    data_loader = DatasetGeneratedDataLoader(dataset, batch_size=batch_size, shuffle=False)

    pgd_config = PGDReconLossConfig(
        init="random",
        step_size=0.1,
        n_steps=20,
        mask_scope=mask_scope,
    )

    pgd_recon = PGDReconLoss(
        model=model,
        device=device,
        output_loss_type=config.output_loss_type,
        pgd_config=pgd_config,
        use_delta_component=config.use_delta_component,
    )
    data_loader_iter = iter(data_loader)
    weight_deltas = model.calc_weight_deltas()
    for _ in range(n_batches):
        batch = extract_batch_data(next(data_loader_iter)).to(device)
        target_model_output: OutputWithCache = model(batch, cache_type="input")
        ci = model.calc_causal_importances(
            pre_weight_acts=target_model_output.cache, detach_inputs=False, sampling=config.sampling
        )
        pgd_recon.update(
            batch=batch,
            target_out=target_model_output.output,
            ci=ci,
            weight_deltas=weight_deltas,
            _tokenizer=None,
        )
    loss = pgd_recon.compute().item()
    print(f"{mask_scope}, {seed=}, {n_batches=}, {batch_size=}, {loss=}")
