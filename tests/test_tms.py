from typing import cast

import torch
from torch import nn

from spd.configs import Config
from spd.experiments.tms.configs import TMSModelConfig, TMSTaskConfig, TMSTrainConfig
from spd.experiments.tms.models import TMSModel
from spd.experiments.tms.train_tms import get_model_and_dataloader, train
from spd.identity_insertion import insert_identity_operations_
from spd.registry import get_experiment_config_file_contents
from spd.run_spd import optimize
from spd.utils.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset
from spd.utils.general_utils import set_seed
from spd.utils.run_utils import apply_nested_updates


def test_tms_decomposition_happy_path() -> None:
    """Test that SPD decomposition works on a TMS model."""
    set_seed(0)
    device = "cpu"

    # Load default config from tms_5-2 and apply test overrides
    base_config = get_experiment_config_file_contents("tms_5-2")
    test_overrides = {
        "wandb_project": None,
        "C": 10,
        "steps": 3,
        "batch_size": 4,
        "eval_batch_size": 4,
        "train_log_freq": 2,
        "n_examples_until_dead": 8,  # train_log_freq * batch_size
        "faithfulness_warmup_steps": 2,
        "target_module_patterns": ["linear1", "linear2", "hidden_layers.0"],
        "identity_module_patterns": ["linear1"],
    }
    config_dict = apply_nested_updates(base_config, test_overrides)
    config = Config.model_validate(config_dict)

    tms_model_config = TMSModelConfig(
        n_features=5,
        n_hidden=2,
        n_hidden_layers=1,
        tied_weights=True,
        init_bias_to_zero=False,
        device=device,
    )

    target_model = TMSModel(config=tms_model_config).to(device)
    target_model.eval()

    if config.identity_module_patterns is not None:
        insert_identity_operations_(target_model, identity_patterns=config.identity_module_patterns)

    assert isinstance(config.task_config, TMSTaskConfig)
    dataset = SparseFeatureDataset(
        n_features=target_model.config.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        data_generation_type=config.task_config.data_generation_type,
        value_range=(0.0, 1.0),
        synced_inputs=None,
    )

    train_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.microbatch_size, shuffle=False
    )
    eval_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.microbatch_size, shuffle=False
    )

    tied_weights = None
    if target_model.config.tied_weights:
        tied_weights = [("linear1", "linear2")]

    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=config.n_eval_steps,
        out_dir=None,
        tied_weights=tied_weights,
    )

    assert True, "Test completed successfully"


def test_train_tms_happy_path():
    """Test training a TMS model from scratch."""
    device = "cpu"
    set_seed(0)
    # Set up a small configuration
    config = TMSTrainConfig(
        tms_model_config=TMSModelConfig(
            n_features=3,
            n_hidden=2,
            n_hidden_layers=0,
            tied_weights=False,
            init_bias_to_zero=False,
            device=device,
        ),
        feature_probability=0.1,
        batch_size=32,
        steps=5,
        lr=5e-3,
        data_generation_type="at_least_zero_active",
        fixed_identity_hidden_layers=False,
        fixed_random_hidden_layers=False,
    )

    model, dataloader = get_model_and_dataloader(config, device)

    train(
        model,
        dataloader,
        importance=1.0,
        lr=config.lr,
        lr_schedule=config.lr_schedule,
        steps=config.steps,
        print_freq=1000,
        log_wandb=False,
    )

    assert True, "Test completed successfully"


def test_tms_train_fixed_identity():
    """Check that hidden layer is identity before and after training."""
    device = "cpu"
    set_seed(0)
    config = TMSTrainConfig(
        tms_model_config=TMSModelConfig(
            n_features=3,
            n_hidden=2,
            n_hidden_layers=2,
            tied_weights=False,
            init_bias_to_zero=False,
            device=device,
        ),
        feature_probability=0.1,
        batch_size=32,
        steps=2,
        lr=5e-3,
        data_generation_type="at_least_zero_active",
        fixed_identity_hidden_layers=True,
        fixed_random_hidden_layers=False,
    )

    model, dataloader = get_model_and_dataloader(config, device)

    eye = torch.eye(config.tms_model_config.n_hidden, device=device)

    assert model.hidden_layers is not None
    # Assert that this is an identity matrix
    initial_hidden = cast(nn.Linear, model.hidden_layers[0]).weight.data.clone()
    assert torch.allclose(initial_hidden, eye), "Initial hidden layer is not identity"

    train(
        model,
        dataloader,
        importance=1.0,
        lr=config.lr,
        lr_schedule=config.lr_schedule,
        steps=config.steps,
        print_freq=1000,
        log_wandb=False,
    )

    # Assert that the hidden layers remains identity
    assert torch.allclose(cast(nn.Linear, model.hidden_layers[0]).weight.data, eye), (
        "Hidden layer changed"
    )


def test_tms_train_fixed_random():
    """Check that hidden layer is random before and after training."""
    device = "cpu"
    set_seed(0)
    config = TMSTrainConfig(
        tms_model_config=TMSModelConfig(
            n_features=3,
            n_hidden=2,
            n_hidden_layers=2,
            tied_weights=False,
            init_bias_to_zero=False,
            device=device,
        ),
        feature_probability=0.1,
        batch_size=32,
        steps=2,
        lr=5e-3,
        data_generation_type="at_least_zero_active",
        fixed_identity_hidden_layers=False,
        fixed_random_hidden_layers=True,
    )

    model, dataloader = get_model_and_dataloader(config, device)

    assert model.hidden_layers is not None
    initial_hidden = cast(nn.Linear, model.hidden_layers[0]).weight.data.clone()

    train(
        model,
        dataloader,
        importance=1.0,
        lr=config.lr,
        lr_schedule=config.lr_schedule,
        steps=config.steps,
        print_freq=1000,
        log_wandb=False,
    )

    # Assert that the hidden layers are unchanged
    assert torch.allclose(cast(nn.Linear, model.hidden_layers[0]).weight.data, initial_hidden), (
        "Hidden layer changed"
    )
