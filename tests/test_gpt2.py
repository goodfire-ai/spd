from transformers import PreTrainedModel

from spd.configs import Config, EvalMetricConfig
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.run_spd import optimize
from spd.utils.general_utils import resolve_class, set_seed


def test_gpt_2_decomposition_happy_path() -> None:
    """Test that SPD decomposition works on for GPT-2"""
    set_seed(0)
    device = "cpu"

    # Create config similar to the gpt-2 config in gpt2_config.yaml
    config = Config(
        # WandB
        wandb_project=None,  # Disable wandb for testing
        wandb_run_name=None,
        wandb_run_name_prefix="",
        # General
        seed=0,
        C=10,  # Smaller C for faster testing
        n_mask_samples=1,
        gate_type="vector_mlp",
        gate_hidden_dims=[128],
        target_module_patterns=["transformer.h.*.attn.c_attn", "transformer.h.*.attn.c_proj"],
        # Loss Coefficients
        faithfulness_coeff=200,
        stochastic_recon_coeff=1.0,
        recon_layerwise_coeff=None,
        stochastic_recon_layerwise_coeff=1.0,
        importance_minimality_coeff=1e-2,
        schatten_coeff=None,
        embedding_recon_coeff=None,
        is_embed_unembed_recon=False,
        pnorm=0.9,
        output_loss_type="kl",
        # Training
        lr=1e-3,
        batch_size=4,
        steps=10,  # Run more steps to see improvement
        lr_schedule="cosine",
        lr_exponential_halflife=None,
        lr_warmup_pct=0.01,
        n_eval_steps=1,
        # Logging & Saving
        train_log_freq=50,  # Print at step 0, 50, and 100
        eval_freq=500,
        eval_batch_size=1,
        slow_eval_freq=500,
        slow_eval_on_first_step=True,
        save_freq=None,
        ci_alive_threshold=0.1,
        n_examples_until_dead=200,  # print_freq * batch_size = 50 * 4
        eval_metrics=[
            EvalMetricConfig(classname="CIHistograms", extra_init_kwargs={"n_batches_accum": 5}),
            EvalMetricConfig(classname="ComponentActivationDensity"),
            EvalMetricConfig(classname="CI_L0"),
            EvalMetricConfig(
                classname="CEandKLLosses", extra_init_kwargs={"rounding_threshold": 0.0}
            ),
        ],
        # Pretrained model info
        pretrained_model_class="transformers.GPT2LMHeadModel",
        pretrained_model_path=None,
        pretrained_model_name_hf="SimpleStories/test-SimpleStories-gpt2-1.25M",
        pretrained_model_output_attr="logits",
        tokenizer_name="SimpleStories/test-SimpleStories-gpt2-1.25M",
        # Task Specific
        task_config=LMTaskConfig(
            task_name="lm",
            max_seq_len=16,
            buffer_size=1000,
            dataset_name="roneneldan/TinyStories",
            column_name="text",
            train_data_split="train[:100]",
            eval_data_split="validation[:100]",
        ),
    )

    assert isinstance(config.task_config, LMTaskConfig), "task_config not LMTaskConfig"

    # Create a GPT-2 model
    hf_model_class = resolve_class(config.pretrained_model_class)
    assert issubclass(hf_model_class, PreTrainedModel), (
        f"Model class {hf_model_class} should be a subclass of PreTrainedModel which "
        "defines a `from_pretrained` method"
    )
    assert config.pretrained_model_name_hf is not None
    target_model = hf_model_class.from_pretrained(config.pretrained_model_name_hf)
    target_model.eval()

    train_data_config = DatasetConfig(
        name=config.task_config.dataset_name,
        hf_tokenizer_path=config.pretrained_model_name_hf,
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
        hf_tokenizer_path=config.pretrained_model_name_hf,
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

    # Run optimize function
    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=config.n_eval_steps,
        out_dir=None,
    )

    # Basic assertion to ensure the test ran
    assert True, "Test completed successfully"
