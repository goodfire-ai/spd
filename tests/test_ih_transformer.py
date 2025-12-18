import pytest
import yaml

from spd.configs import Config
from spd.experiments.ih.configs import InductionModelConfig
from spd.experiments.ih.model import InductionTransformer
from spd.identity_insertion import insert_identity_operations_
from spd.run_spd import optimize
from spd.settings import REPO_ROOT
from spd.utils.data_utils import DatasetGeneratedDataLoader, InductionDataset
from spd.utils.general_utils import set_seed
from spd.utils.run_utils import apply_nested_updates


@pytest.mark.slow
def test_ih_transformer_decomposition_happy_path() -> None:
    """Test that SPD decomposition works on a 2-layer, 1 head attention-only Transformer model.

    TODO: Use a real pretrained_model_path in the config instead of randomly initializing one.
    """
    set_seed(0)
    device = "cpu"

    config_path = REPO_ROOT / "spd/experiments/ih/ih_config.yaml"
    base_config_dict = yaml.safe_load(config_path.read_text())
    test_overrides = {
        "wandb_project": None,
        "C": 10,
        "steps": 2,
        "batch_size": 4,
        "eval_batch_size": 1,
        "train_log_freq": 50,
        "n_examples_until_dead": 999,
        "pretrained_model_path": None,
        "n_eval_steps": 1,
    }
    config_dict = apply_nested_updates(base_config_dict, test_overrides)
    config = Config(**config_dict)

    ih_transformer_config = InductionModelConfig(
        vocab_size=128,
        d_model=16,
        n_layers=2,
        n_heads=1,
        seq_len=64,
        use_ff=False,
        use_pos_encoding=True,
        use_layer_norm=False,
        ff_fanout=4,
    )

    target_model = InductionTransformer(ih_transformer_config).to(device)
    target_model.eval()
    target_model.requires_grad_(False)

    if config.identity_module_patterns is not None:
        insert_identity_operations_(target_model, identity_patterns=config.identity_module_patterns)

    dataset = InductionDataset(
        seq_len=ih_transformer_config.seq_len,
        vocab_size=ih_transformer_config.vocab_size,
        device=device,
        prefix_window=ih_transformer_config.seq_len - 3,
    )

    train_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.microbatch_size, shuffle=False
    )
    eval_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.microbatch_size, shuffle=False
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
