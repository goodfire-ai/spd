# %%
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from spd.data import DatasetConfig, create_data_loader, loop_dataloader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import replace_pydantic_model, set_seed

# %%

set_seed(0)

device = get_device()
run_info = SPDRunInfo.from_path("wandb:goodfire/spd/runs/9d313yrl")
config = run_info.config
assert isinstance(config.task_config, LMTaskConfig), "task_config not LMTaskConfig"

task_config = replace_pydantic_model(
    config.task_config, {"max_seq_len": 512, "train_data_split": "train[:1000]"}
)

model = ComponentModel.from_run_info(run_info)
model.to(device)
model.eval()

# Set the target model params to frozen
model.target_model.requires_grad_(False)

train_data_config = DatasetConfig(
    name=task_config.dataset_name,
    hf_tokenizer_path=config.tokenizer_name,
    split=task_config.train_data_split,
    n_ctx=task_config.max_seq_len,
    is_tokenized=task_config.is_tokenized,
    streaming=task_config.streaming,
    column_name=task_config.column_name,
    shuffle_each_epoch=task_config.shuffle_each_epoch,
    seed=None,
)

# %%
batch_size = 10000
set_seed(0)
data_loader, _tokenizer = create_data_loader(
    dataset_config=train_data_config,
    batch_size=batch_size,
    buffer_size=task_config.buffer_size,
    global_seed=config.seed,
    ddp_rank=0,
    ddp_world_size=1,
)
data_iterator = loop_dataloader(data_loader)


# %%
import torch
import torch.nn.functional as F

# Get a batch
batch = next(data_iterator)
tokens = batch["tokens"].to(device)
print(f"Batch shape: {tokens.shape}")

# Forward pass with original model
with torch.no_grad():
    original_logits = model.target_model(tokens)
    original_log_probs = F.log_softmax(original_logits, dim=-1)
    print(f"Original logits shape: {original_logits.shape}")

# Zero out the embedding matrix
embedding_layer = model.target_model.embed
original_weight = embedding_layer.weight.data.clone()
print(f"Embedding shape: {embedding_layer.weight.shape}")

embedding_layer.weight.data.zero_()

# Forward pass with zeroed embedding
with torch.no_grad():
    ablated_logits = model.target_model(tokens)
    ablated_log_probs = F.log_softmax(ablated_logits, dim=-1)

# Restore original embedding
embedding_layer.weight.data = original_weight

# Calculate KL divergence
# KL(P || Q) where P is original, Q is ablated
# F.kl_div expects log-probabilities for Q and probabilities for P
original_probs = torch.exp(original_log_probs)
kl_div = F.kl_div(ablated_log_probs, original_probs, reduction="batchmean")

print(f"\nKL Divergence (original || ablated): {kl_div.item():.4f}")

# Also compute in the other direction for symmetry
ablated_probs = torch.exp(ablated_log_probs)
kl_div_reverse = F.kl_div(original_log_probs, ablated_probs, reduction="batchmean")
print(f"KL Divergence (ablated || original): {kl_div_reverse.item():.4f}")

# Compute mean KL per token position
kl_per_position = F.kl_div(
    ablated_log_probs, original_probs, reduction="none"
).sum(dim=-1)
print(f"Mean KL per position: {kl_per_position.mean().item():.4f}")
print(f"Max KL per position: {kl_per_position.max().item():.4f}")


# %%
