"""Profile the eval phase of pile_llama_simple_mlp-4L to find slow metrics.

Instruments the evaluate() function with per-metric timing.
Runs a single step and reports detailed timing breakdown.
"""

import time
from pathlib import Path

import torch

from spd.configs import (
    Config,
    LMTaskConfig,
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
    PGDMultiBatchConfig,
)
from spd.data import DatasetConfig, create_data_loader, loop_dataloader
from spd.eval import init_metric
from spd.models.component_model import ComponentModel
from spd.persistent_pgd import PersistentPGDState
from spd.pretrain.run_info import PretrainRunInfo
from spd.run_spd import get_unique_metric_configs, run_faithfulness_warmup
from spd.utils.general_utils import (
    bf16_autocast,
    extract_batch_data,
    resolve_class,
    set_seed,
)
from spd.utils.module_utils import expand_module_patterns


def main() -> None:
    config_path = Path("spd/experiments/lm/pile_llama_simple_mlp-4L.yaml")
    config = Config.from_file(config_path)

    # Override for profiling
    config_dict = config.model_dump(mode="json")
    config_dict["wandb_project"] = None
    config_dict["steps"] = 1
    config_dict["eval_freq"] = 1
    config_dict["train_log_freq"] = 1
    config_dict["save_freq"] = None
    config_dict["batch_size"] = 8
    config_dict["eval_batch_size"] = 32
    config = Config(**config_dict)

    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pretrained model
    t0 = time.time()
    pretrained_model_class = resolve_class(config.pretrained_model_class)
    assert config.pretrained_model_name is not None
    run_info = PretrainRunInfo.from_path(config.pretrained_model_name)
    if "model_type" not in run_info.model_config_dict:
        run_info.model_config_dict["model_type"] = config.pretrained_model_class.split(".")[-1]
    target_model = pretrained_model_class.from_run_info(run_info)
    target_model.eval()
    print(f"Model loading: {time.time() - t0:.1f}s")

    # Load data
    assert isinstance(config.task_config, LMTaskConfig)
    t0 = time.time()
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
    train_loader, _ = create_data_loader(
        dataset_config=train_data_config,
        batch_size=config.microbatch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed,
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
    eval_loader, _ = create_data_loader(
        dataset_config=eval_data_config,
        batch_size=config.eval_batch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed + 1,
    )
    print(f"Data loading: {time.time() - t0:.1f}s")

    train_iterator = loop_dataloader(train_loader)
    eval_iterator = loop_dataloader(eval_loader)

    # Setup ComponentModel
    t0 = time.time()
    target_model.requires_grad_(False)
    module_path_info = expand_module_patterns(target_model, config.all_module_info)
    model = ComponentModel(
        target_model=target_model,
        module_path_info=module_path_info,
        ci_config=config.ci_config,
        sigmoid_type=config.sigmoid_type,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
    )
    model.to(device)
    print(f"ComponentModel setup: {time.time() - t0:.1f}s")

    # Get component params
    component_params = []
    for name in model.target_module_paths:
        component_params.extend(model.components[name].parameters())

    # Faithfulness warmup
    t0 = time.time()
    if config.faithfulness_warmup_steps > 0:
        run_faithfulness_warmup(model, component_params, config)
    print(
        f"Faithfulness warmup ({config.faithfulness_warmup_steps} steps): {time.time() - t0:.1f}s"
    )

    # Setup eval metrics
    eval_metric_configs = get_unique_metric_configs(
        loss_configs=config.loss_metric_configs,
        eval_configs=config.eval_metric_configs,
    )

    multibatch_pgd_eval_configs = [
        cfg for cfg in eval_metric_configs if isinstance(cfg, PGDMultiBatchConfig)
    ]
    eval_metric_configs = [
        cfg for cfg in eval_metric_configs if cfg not in multibatch_pgd_eval_configs
    ]

    # PersistentPGD state
    persistent_pgd_configs = [
        cfg
        for cfg in config.loss_metric_configs
        if isinstance(cfg, PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig)
    ]

    sample_batch = extract_batch_data(next(train_iterator))
    batch_dims = sample_batch.shape if config.output_loss_type == "kl" else sample_batch.shape[:-1]

    ppgd_states = {
        ppgd_cfg: PersistentPGDState(
            module_to_c=model.module_to_c,
            seq_len=batch_dims[-1],
            device=device,
            use_delta_component=config.use_delta_component,
            cfg=ppgd_cfg,
        )
        for ppgd_cfg in persistent_pgd_configs
    }
    ppgd_maskss = {cfg: ppgd_states[cfg].masks for cfg in persistent_pgd_configs}

    # Print memory before eval
    if torch.cuda.is_available():
        print("\nGPU memory before eval:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GiB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GiB")

    # ============================
    # PROFILE EVAL
    # ============================
    print("\n" + "=" * 60)
    print("PROFILING EVAL (slow_step=True, like step 0)")
    print("=" * 60)

    slow_step = True

    with torch.no_grad(), bf16_autocast(enabled=config.autocast_bf16):
        # 1. Fetch eval batch
        t0 = time.time()
        batch_raw = next(eval_iterator)
        batch = extract_batch_data(batch_raw).to(device)
        print(f"\nFetching eval batch: {time.time() - t0:.3f}s (shape: {batch.shape})")

        # 2. Target model forward + CI
        t0 = time.time()
        target_output = model(batch, cache_type="input")
        ci = model.calc_causal_importances(
            pre_weight_acts=target_output.cache,
            detach_inputs=False,
            sampling=config.sampling,
        )
        torch.cuda.synchronize()
        print(f"Target forward + CI: {time.time() - t0:.3f}s")

        # 3. Weight deltas
        t0 = time.time()
        weight_deltas = model.calc_weight_deltas()
        torch.cuda.synchronize()
        print(f"Weight deltas: {time.time() - t0:.3f}s")

        # 4. Per-metric timing
        print(f"\nPer-metric timing (eval_batch_size={config.eval_batch_size}):")
        print("-" * 60)

        for cfg in eval_metric_configs:
            metric = init_metric(
                cfg=cfg,
                model=model,
                ppgd_maskss=ppgd_maskss,
                run_config=config,
                device=device,
            )
            if metric.slow and not slow_step:
                print(f"  {type(metric).__name__:40s} SKIPPED (slow)")
                continue

            t0 = time.time()
            metric.update(
                batch=batch,
                target_out=target_output.output,
                pre_weight_acts=target_output.cache,
                ci=ci,
                current_frac_of_training=0.0,
                weight_deltas=weight_deltas,
            )
            torch.cuda.synchronize()
            update_time = time.time() - t0

            t0 = time.time()
            result = metric.compute()
            torch.cuda.synchronize()
            compute_time = time.time() - t0

            total_time = update_time + compute_time
            is_slow = " [SLOW METRIC]" if metric.slow else ""
            print(
                f"  {type(metric).__name__:40s} "
                f"update={update_time:.3f}s  compute={compute_time:.3f}s  "
                f"total={total_time:.3f}s{is_slow}"
            )

            # Memory after each metric
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"    Memory: allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB")

            del metric, result

    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)

    if torch.cuda.is_available():
        print("\nFinal GPU memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GiB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GiB")
        print(f"  Peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GiB")


if __name__ == "__main__":
    main()
