import pytest

from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.identity_insertion import insert_identity_operations_
from spd.registry import get_experiment_config_file_contents
from spd.run_spd import optimize
from spd.utils.general_utils import resolve_class, set_seed
from spd.utils.run_utils import apply_nested_updates

# Config-specific test parameters for different GPT2 configurations
GPT2_CONFIG_PARAMS = {
    "ss_gpt2_simple": {
        # Uses simple_stories_train.models.gpt2_simple.GPT2Simple (wandb-hosted model)
        "target_module_patterns": ["h.2.attn.q_proj", "h.3.mlp.c_fc"],
        "identity_module_patterns": ["h.1.attn.q_proj"],
    },
    "ss_gpt2": {
        # Uses transformers.GPT2LMHeadModel (HuggingFace transformers library)
        "target_module_patterns": ["transformer.h.1.mlp.c_fc"],
        "identity_module_patterns": None,
    },
}


@pytest.mark.slow
@pytest.mark.parametrize("experiment_name", ["ss_gpt2_simple", "ss_gpt2"])
def test_gpt2_decomposition_happy_path(experiment_name: str) -> None:
    """Test that SPD decomposition works on different GPT-2 configurations.

    Tests both:
    - ss_gpt2_simple: Uses simple_stories_train GPT2Simple model (wandb-hosted)
    - ss_gpt2: Uses transformers.GPT2LMHeadModel (HuggingFace transformers library)
    """
    set_seed(0)
    device = "cpu"

    config_params = GPT2_CONFIG_PARAMS[experiment_name]
    base_config_dict = get_experiment_config_file_contents(experiment_name)
    test_overrides = {
        "wandb_project": None,
        "C": 10,
        "steps": 2,
        "batch_size": 4,
        "eval_batch_size": 1,
        "train_log_freq": 50,
        "n_examples_until_dead": 999,
        "task_config.max_seq_len": 8,
        "task_config.train_data_split": "train[:100]",
        "task_config.eval_data_split": "test[100:200]",
        "target_module_patterns": config_params["target_module_patterns"],
        "identity_module_patterns": config_params["identity_module_patterns"],
        "eval_metric_configs": [],  # Disable eval metrics to avoid layer matching issues
    }
    config_dict = apply_nested_updates(base_config_dict, test_overrides)
    config = Config(**config_dict)

    assert isinstance(config.task_config, LMTaskConfig), "task_config not LMTaskConfig"
    pretrained_model_class = resolve_class(config.pretrained_model_class)
    assert hasattr(pretrained_model_class, "from_pretrained"), (
        f"Model class {pretrained_model_class} should have a `from_pretrained` method"
    )
    assert config.pretrained_model_name is not None

    # Handle simple_stories_train models specially (they use from_run_info)
    if config.pretrained_model_class.startswith("simple_stories_train"):
        from simple_stories_train.run_info import RunInfo as SSRunInfo

        run_info = SSRunInfo.from_path(config.pretrained_model_name)
        assert hasattr(pretrained_model_class, "from_run_info")
        target_model = pretrained_model_class.from_run_info(run_info)  # pyright: ignore[reportAttributeAccessIssue]
    else:
        target_model = pretrained_model_class.from_pretrained(config.pretrained_model_name)  # pyright: ignore[reportAttributeAccessIssue]
    target_model.eval()

    if config.identity_module_patterns is not None:
        insert_identity_operations_(target_model, identity_patterns=config.identity_module_patterns)

    train_data_config = DatasetConfig(
        name=config.task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=config.task_config.train_data_split,
        n_ctx=config.task_config.max_seq_len,
        is_tokenized=config.task_config.is_tokenized,
        streaming=config.task_config.streaming,
        column_name=config.task_config.column_name,
        seed=None,
    )

    train_loader, _tokenizer = create_data_loader(
        dataset_config=train_data_config,
        batch_size=config.batch_size,
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
        seed=None,
    )
    eval_loader, _ = create_data_loader(
        dataset_config=eval_data_config,
        batch_size=config.batch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed + 1,
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
