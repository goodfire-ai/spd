from spd.configs import PGDReconLossConfig
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.metrics.pgd_masked_recon_loss import PGDReconLoss
from spd.models.component_model import ComponentModel, OutputWithCache, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import extract_batch_data, replace_pydantic_model, set_seed

device = get_device()
run_info = SPDRunInfo.from_path("wandb:goodfire/spd/runs/lxs77xye")
config = run_info.config
assert isinstance(config.task_config, LMTaskConfig), "task_config not LMTaskConfig"


batch_size = 2
max_seq_len = 8
n_batches = 20
seed = 1
mask_scope = "shared_across_batch"

for n_batches in [1, 2, 4, 8, 16, 32, 64]:
    set_seed(seed)
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

    pgd_config = PGDReconLossConfig(
        init="random",
        step_size=0.1,
        n_steps=20,
        mask_scope=mask_scope,
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
            pre_weight_acts=target_model_output.cache,
            detach_inputs=False,
            sampling=config.sampling,
        )
        pgd_recon.update(
            batch=batch,
            target_out=target_model_output.output,
            ci=ci,
            weight_deltas=weight_deltas,
        )

    loss = pgd_recon.compute().item()
    print(f"n_batches: {n_batches}, batch_size: {batch_size}, seq_len: {max_seq_len}, Loss: {loss}")
