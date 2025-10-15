# %%
from spd.configs import (
    CIMeanPerComponentConfig,
    ComponentActivationDensityConfig,
    CEandKLLossesConfig,
    Config,
    FaithfulnessLossTrainConfig,
    ImportanceMinimalityLossTrainConfig,
    PGDReconSubsetLossTrainConfig,
    StochasticReconLayerwiseLossTrainConfig,
    StochasticReconSubsetLossTrainConfig,
    CIHistogramsConfig
)
from spd.data import DatasetConfig, create_data_loader, loop_dataloader
from spd.experiments.lm.configs import LMTaskConfig
from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.utils.distributed_utils import get_device, is_main_process
from spd.utils.general_utils import extract_batch_data, load_config, resolve_class

# %%

path = "spd/experiments/lm/ss_llama_config.yaml"

config = load_config(path, config_model=Config)

device = get_device()
assert isinstance(config.task_config, LMTaskConfig), "task_config not LMTaskConfig"

pretrained_model_class = resolve_class(config.pretrained_model_class)
assert hasattr(pretrained_model_class, "from_pretrained"), (
    f"Model class {pretrained_model_class} should have a `from_pretrained` method"
)
assert config.pretrained_model_name is not None
# %%
target_model = pretrained_model_class.from_pretrained(config.pretrained_model_name)  # pyright: ignore[reportAttributeAccessIssue]

target_model.eval()
# %%

if is_main_process():
    logger.info("Loading dataset...")
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

# %%

train_loader, _tokenizer = create_data_loader(
    dataset_config=train_data_config,
    batch_size=config.batch_size,
    buffer_size=config.task_config.buffer_size,
    global_seed=config.seed,
    ddp_rank=0,
    ddp_world_size=1,
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

# %%

eval_loader, _ = create_data_loader(
    dataset_config=eval_data_config,
    batch_size=config.eval_batch_size,
    buffer_size=config.task_config.buffer_size,
    global_seed=config.seed + 1,
    ddp_rank=0,
    ddp_world_size=1,
)

# %%

target_model.requires_grad_(False)

model = ComponentModel(
    target_model=target_model,
    target_module_patterns=config.all_module_patterns,
    C=config.C,
    ci_fn_type=config.ci_fn_type,
    ci_fn_hidden_dims=config.ci_fn_hidden_dims,
    pretrained_model_output_attr=config.pretrained_model_output_attr,
)
model.to(device)

# %%

dl_iter = loop_dataloader(train_loader)


def b():
    return extract_batch_data(next(dl_iter)).to(device)


# %%

b().shape
# %%

# %%

"""
the existing config (old schema):

Config(
    wandb_project= "spd",
    wandb_run_name= None,
    wandb_run_name_prefix= "",
    seed= 0,
    C= 5000,
    n_mask_samples= 1,
    gate_type= "mlp",
    gate_hidden_dims= [16],
    sampling= "continuous",
    sigmoid_type= "leaky_hard",
    target_module_patterns= ["model.embed_tokens"],
    identity_module_patterns= None,
    use_delta_component= true,
    faithfulness_coeff= None,
    ci_recon_coeff= None,
    stochastic_recon_coeff= None,
    ci_recon_layerwise_coeff= None,
    stochastic_recon_layerwise_coeff= None,
    ci_masked_recon_subset_coeff= None,
    stochastic_recon_subset_coeff= 1.0,
    importance_minimality_coeff= 0.05,
    importance_minimality_coeff_start_frac= 0.0,
    importance_minimality_coeff_final= 0.05,
    importance_minimality_coeff_end_frac= 0.5,
    pnorm= 1.0,
    p_anneal_start_frac= 1.0,
    p_anneal_final_p= None,
    p_anneal_end_frac= 1.0,
    output_loss_type= "kl",
    lr= 0.0005,
    steps= 500000,
    batch_size= 256,
    gradient_accumulation_steps= 1,
    lr_schedule= "constant",
    lr_exponential_halflife= None,
    lr_warmup_pct= 0.0,
    pgd_mask_enabled= true,
    pgd_mask_steps= 2,
    pgd_mask_step_size= 0.5,
    pgd_mask_random_init= true,
    adv_mix_start_frac= 0.0,
    adv_mix_end_frac= 1.0,
    adv_mix_adv_weight_start= 0.1,
    adv_mix_adv_weight_end= 0.1,
    adv_mix_rand_weight_start= 0.9,
    adv_mix_rand_weight_end= 0.9,
    out_dir= None,
    train_log_freq= 200,
    eval_freq= 1000,
    eval_batch_size= 256,
    slow_eval_freq= 2000,
    n_eval_steps= 5,
    slow_eval_on_first_step= true,
    save_freq= None,
    eval_metrics= [
        CIHistogramsConfig(classname= "CIHistograms", extra_init_kwargs= {n_batches_accum= 5}},
        {classname= "ComponentActivationDensity", extra_init_kwargs= {}},
        {classname= "CEandKLLosses", extra_init_kwargs= {rounding_threshold= 0.0}},
        {classname= "CIMeanPerComponent", extra_init_kwargs= {}},
        {classname= "FaithfulnessLoss", extra_init_kwargs= {}}
    ],
    ci_alive_threshold= 0.0,
    n_examples_until_dead= 1368400,
    pretrained_model_class= "transformers.LlamaForCausalLM",
    pretrained_model_path= None,
    pretrained_model_name= "SimpleStories/SimpleStories-1.25M",
    pretrained_model_output_attr= "logits",
    tokenizer_name= "SimpleStories/SimpleStories-1.25M",
    task_config= LMTaskConfig(
        task_name= "lm",
        max_seq_len= 512,
        buffer_size= 1000,
        dataset_name= "SimpleStories/SimpleStories",
        column_name= "story",
        train_data_split= "train",
        eval_data_split= "test",
        shuffle_each_epoch= True,
        is_tokenized= False,
        streaming= False,
    ),
    dist_backend= None,
    )
"""

# new one:

Config(
    wandb_project="spd",
    wandb_run_name=None,
    wandb_run_name_prefix="",
    seed=0,
    C=5000,
    n_mask_samples=1,
    ci_fn_type="mlp",
    ci_fn_hidden_dims=[16],
    sampling="continuous",
    sigmoid_type="leaky_hard",
    target_module_patterns=["model.embed_tokens"],
    identity_module_patterns=None,
    use_delta_component=True,
    loss_metric_configs=[
        ImportanceMinimalityLossTrainConfig(
            coeff=0.05,
            pnorm=1.0,
            p_anneal_start_frac=0.0,
            p_anneal_final_p=None,
            p_anneal_end_frac=1.0,
        ),
        StochasticReconSubsetLossTrainConfig(coeff=1.0),
        PGDReconSubsetLossTrainConfig(
            coeff=0.02,
            step_size=0.01,
            n_steps=2,
            mask_scope="unique_per_datapoint",
            init="random",
        ),
    ],
    output_loss_type="kl",
    lr=0.0005,
    steps=500000,
    batch_size=256,
    gradient_accumulation_steps=1,
    lr_schedule="constant",
    lr_exponential_halflife=None,
    lr_warmup_pct=0.0,
    out_dir=None,
    train_log_freq=200,
    eval_freq=1000,
    eval_batch_size=256,
    slow_eval_freq=2000,
    n_eval_steps=5,
    slow_eval_on_first_step=True,
    save_freq=None,
    eval_metric_configs=[
        CIHistogramsConfig(n_batches_accum=5),
        ComponentActivationDensityConfig(),
        CEandKLLossesConfig(rounding_threshold= 0.0),
        CIMeanPerComponentConfig(),
        FaithfulnessLossTrainConfig(),
    ],
    ci_alive_threshold=0.0,
    n_examples_until_dead=1368400,
    pretrained_model_class="transformers.LlamaForCausalLM",
    pretrained_model_path=None,
    pretrained_model_name="SimpleStories/SimpleStories-1.25M",
    pretrained_model_output_attr="logits",
    tokenizer_name="SimpleStories/SimpleStories-1.25M",
    task_config=LMTaskConfig(
        task_name="lm",
        max_seq_len=512,
        buffer_size=1000,
        dataset_name="SimpleStories/SimpleStories",
        column_name="story",
        train_data_split="train",
        eval_data_split="test",
        shuffle_each_epoch=True,
        is_tokenized=False,
        streaming=False,
    ),
)

