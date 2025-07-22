from spd.configs import Config, FiguresFnConfig, MetricsFnConfig, ResidualMLPTaskConfig
from spd.experiments.resid_mlp.models import ResidualMLP, ResidualMLPConfig
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.run_spd import optimize
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.general_utils import set_seed


def test_resid_mlp_decomposition_happy_path() -> None:
    """Test that SPD decomposition works on a 2-layer ResidualMLP model."""
    set_seed(0)
    device = "cpu"

    # Create a 2-layer ResidualMLP config
    resid_mlp_config = ResidualMLPConfig(
        n_features=5,
        d_embed=4,
        d_mlp=6,
        n_layers=2,
        act_fn_name="relu",
        in_bias=True,
        out_bias=True,
    )

    # Create config similar to the 2-layer config in resid_mlp_config.yaml
    config = Config(
        # WandB
        wandb_project=None,  # Disable wandb for testing
        wandb_run_name=None,
        wandb_run_name_prefix="",
        # General
        seed=0,
        C=10,  # Smaller C for faster testing
        n_mask_samples=1,
        gate_type="mlp",
        gate_hidden_dims=[8],
        target_module_patterns=[
            "layers.*.mlp_in",
            "layers.*.mlp_out",
        ],
        # Loss Coefficients
        faithfulness_coeff=1.0,
        recon_coeff=2.0,
        stochastic_recon_coeff=1.0,
        recon_layerwise_coeff=None,
        stochastic_recon_layerwise_coeff=None,
        importance_minimality_coeff=3e-3,
        schatten_coeff=None,
        embedding_recon_coeff=None,
        is_embed_unembed_recon=False,
        pnorm=0.9,
        output_loss_type="mse",
        # Training
        lr=1e-3,
        batch_size=4,
        steps=3,  # Run more steps to see improvement
        lr_schedule="cosine",
        lr_exponential_halflife=None,
        lr_warmup_pct=0.01,
        n_eval_steps=1,
        # Logging & Saving
        image_freq=None,
        image_on_first_step=True,
        print_freq=50,  # Print at step 0, 50, and 100
        save_freq=None,
        figures_fns=[
            FiguresFnConfig(name="ci_histograms"),
            FiguresFnConfig(name="mean_component_activation_counts"),
            FiguresFnConfig(name="uv_and_identity_ci"),
        ],
        metrics_fns=[
            MetricsFnConfig(name="ci_l0"),
        ],
        # Pretrained model info
        pretrained_model_class="spd.experiments.resid_mlp.models.ResidualMLP",
        pretrained_model_path=None,
        pretrained_model_name_hf=None,
        pretrained_model_output_attr=None,
        tokenizer_name=None,
        # Task Specific
        task_config=ResidualMLPTaskConfig(
            task_name="residual_mlp",
            feature_probability=0.01,
            data_generation_type="at_least_zero_active",
        ),
    )

    # Create a pretrained model
    target_model = ResidualMLP(config=resid_mlp_config).to(device)
    target_model.eval()

    assert isinstance(config.task_config, ResidualMLPTaskConfig)
    # Create dataset
    dataset = ResidualMLPDataset(
        n_features=resid_mlp_config.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        calc_labels=False,  # Our labels will be the output of the target model
        label_type=None,
        act_fn_name=None,
        label_fn_seed=None,
        label_coeffs=None,
        data_generation_type=config.task_config.data_generation_type,
        synced_inputs=None,
    )

    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

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
