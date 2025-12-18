from spd.configs import Config
from spd.experiments.resid_mlp.configs import ResidMLPModelConfig, ResidMLPTaskConfig
from spd.experiments.resid_mlp.models import ResidMLP
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
from spd.identity_insertion import insert_identity_operations_
from spd.registry import get_experiment_config_file_contents
from spd.run_spd import optimize
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.general_utils import set_seed
from spd.utils.run_utils import apply_nested_updates


def test_resid_mlp_decomposition_happy_path() -> None:
    """Test that SPD decomposition works on a 2-layer ResidMLP model."""
    set_seed(0)
    device = "cpu"

    base_config_dict = get_experiment_config_file_contents("resid_mlp2")
    test_overrides = {
        "wandb_project": None,
        "C": 10,
        "steps": 3,
        "batch_size": 4,
        "eval_batch_size": 4,
        "train_log_freq": 50,
        "n_examples_until_dead": 999,
        "eval_metric_configs.IdentityCIError.identity_ci": [
            {"layer_pattern": "layers.*.mlp_in", "n_features": 5}
        ],
        "eval_metric_configs.IdentityCIError.dense_ci": [
            {"layer_pattern": "layers.*.mlp_out", "k": 3}
        ],
    }
    config_dict = apply_nested_updates(base_config_dict, test_overrides)
    config = Config(**config_dict)

    resid_mlp_model_config = ResidMLPModelConfig(
        n_features=5,
        d_embed=4,
        d_mlp=6,
        n_layers=2,
        act_fn_name="relu",
        in_bias=True,
        out_bias=True,
    )

    target_model = ResidMLP(config=resid_mlp_model_config).to(device)
    target_model.requires_grad_(False)

    if config.identity_module_patterns is not None:
        insert_identity_operations_(target_model, identity_patterns=config.identity_module_patterns)

    assert isinstance(config.task_config, ResidMLPTaskConfig)
    dataset = ResidMLPDataset(
        n_features=resid_mlp_model_config.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        calc_labels=False,
        label_type=None,
        act_fn_name=None,
        label_fn_seed=None,
        label_coeffs=None,
        data_generation_type=config.task_config.data_generation_type,
        synced_inputs=None,
    )

    train_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.microbatch_size, shuffle=False
    )
    eval_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.eval_batch_size, shuffle=False
    )

    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=config.n_eval_steps,
        out_dir=None,
    )

    assert True, "Test completed successfully"
