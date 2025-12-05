import pytest
from transformers import PreTrainedModel

from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.identity_insertion import insert_identity_operations_
from spd.registry import get_experiment_config_file_contents
from spd.run_spd import optimize
from spd.utils.general_utils import resolve_class, set_seed
from spd.utils.run_utils import apply_nested_updates


@pytest.mark.slow
def test_gpt_2_decomposition_happy_path() -> None:
    """Test that SPD decomposition works on for GPT-2"""
    set_seed(0)
    device = "cpu"

    base_config = get_experiment_config_file_contents("ss_gpt2_simple")
    test_overrides = {
        "wandb_project": None,
        "C": 10,
        "steps": 2,
        "batch_size": 4,
        "eval_batch_size": 1,
        "train_log_freq": 50,
        "n_examples_until_dead": 200,  # train_log_freq * batch_size
        "task_config.max_seq_len": 16,
        "task_config.train_data_split": "train[:100]",
        "task_config.eval_data_split": "test[100:200]",
        "target_module_patterns": ["transformer.h.2.attn.c_attn", "transformer.h.3.mlp.c_fc"],
        "identity_module_patterns": ["transformer.h.1.attn.c_attn"],
    }
    config_dict = apply_nested_updates(base_config, test_overrides)
    config = Config.model_validate(config_dict)

    assert isinstance(config.task_config, LMTaskConfig), "task_config not LMTaskConfig"
    hf_model_class = resolve_class(config.pretrained_model_class)
    assert issubclass(hf_model_class, PreTrainedModel), (
        f"Model class {hf_model_class} should be a subclass of PreTrainedModel which "
        "defines a `from_pretrained` method"
    )
    assert config.pretrained_model_name is not None
    target_model = hf_model_class.from_pretrained(config.pretrained_model_name)
    target_model.eval()

    if config.identity_module_patterns is not None:
        insert_identity_operations_(target_model, identity_patterns=config.identity_module_patterns)

    train_data_config = DatasetConfig(
        name=config.task_config.dataset_name,
        hf_tokenizer_path=config.pretrained_model_name,
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
        hf_tokenizer_path=config.pretrained_model_name,
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

    assert True, "Test completed successfully"
