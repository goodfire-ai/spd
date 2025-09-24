from spd.configs import Config, MetricConfig
from spd.experiments.ih.configs import IHTaskConfig, InductionModelConfig
from spd.experiments.ih.model import InductionTransformer
from spd.identity_insertion import insert_identity_operations_
from spd.run_spd import optimize
from spd.utils.data_utils import DatasetGeneratedDataLoader, InductionDataset
from spd.utils.general_utils import set_seed


def test_ih_transformer_decomposition_happy_path() -> None:
    """Test that SPD decomposition works on a 2-layer, 1 head attention-only Transformer model"""
    set_seed(0)
    device = "cpu"

    # Create a 2-layer InductionTransformer config
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

    # Create config similar to the induction_head transformer config in ih_config.yaml
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
        target_module_patterns=["blocks.*.attn.q_proj", "blocks.*.attn.k_proj"],
        identity_module_patterns=["blocks.*.attn.q_proj"],
        # Loss Coefficients
        loss_metric_configs=[
            MetricConfig(
                classname="ImportanceMinimalityLoss",
                coeff=1e-2,
                extra_init_kwargs={"pnorm": 0.9, "eps": 1e-12},
            ),
            MetricConfig(classname="StochasticReconLayerwiseLoss", coeff=1.0),
            MetricConfig(classname="StochasticReconLoss", coeff=1.0),
            MetricConfig(classname="FaithfulnessLoss", coeff=200),
        ],
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
        eval_metric_configs=[
            MetricConfig(classname="CIHistograms"),
            MetricConfig(classname="ComponentActivationDensity"),
            MetricConfig(classname="CI_L0"),
        ],
        # Pretrained model info
        pretrained_model_class="spd.experiments.ih.model.InductionTransformer",
        pretrained_model_path=None,
        pretrained_model_name=None,
        pretrained_model_output_attr=None,
        tokenizer_name=None,
        # Task Specific
        task_config=IHTaskConfig(
            task_name="induction_head",
        ),
    )

    # Create a pretrained model

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
