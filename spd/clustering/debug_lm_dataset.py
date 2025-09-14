"""Debug script for loading LM datasets and getting max activations."""

# %%
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from spd.clustering.activations import component_activations
from spd.models.component_model import ComponentModel, SPDRunInfo

# %% Check what's in the dataset
dataset_name = "lennart-finke/SimpleStories"
ds = load_dataset(dataset_name, split="train", streaming=False)
print(f"{dataset_name = }")
print(f"{ds.features = }")
print(f"{len(ds) = }")
print(f"{ds[0].keys() = }")
print(f"{ds[0] = }")

# %% Check data type
# Get the actual column name
column_names = list(ds[0].keys())
print(f"{column_names = }")
first_column = column_names[0] if column_names else None
print(f"{first_column = }")

if first_column:
    first_item = ds[0][first_column]
    print(f"{type(first_item) = }")
    print(f"{first_item = }")
    if isinstance(first_item, list):
        print(f"{len(first_item) = }")
        print(f"{first_item[:10] = }")

# %% Skip torch format for now since it's text data
print("Skipping torch format since data is text, not tokens")

# %% Test with our actual clustering code

model_path = "wandb:goodfire/spd/runs/okd93sk2"
print(f"{model_path = }")

spd_run = SPDRunInfo.from_path(model_path)
print(f"{spd_run.config.tokenizer_name = }")

model = ComponentModel.from_pretrained(spd_run.checkpoint_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"{device = }")

# %% Get tokenizer from model config and tokenize the stories
tokenizer = AutoTokenizer.from_pretrained(spd_run.config.tokenizer_name)
if first_column:
    # Take a story and tokenize it
    story_text = ds[0][first_column]
    print(f"{story_text[:100] = }")

    # Tokenize the story
    tokens = tokenizer.encode(story_text, max_length=1024, truncation=True, return_tensors="pt")
    print(f"{tokens.shape = }")
    print(f"{tokens = }")

    # Decode to verify
    decoded = tokenizer.decode(tokens[0][:50])
    print(f"{decoded = }")

# %% Test component activations with tokenized data
# Create a batch of tokenized stories
story_batch = []
for i in range(4):  # batch size 4
    story_text = ds[i][first_column]
    tokens = tokenizer.encode(story_text, max_length=64, truncation=True, return_tensors="pt")
    story_batch.append(tokens)

batch_tensor = torch.cat(story_batch, dim=0).to(device)
print(f"{batch_tensor.shape = }")

activations = component_activations(model, device, batch=batch_tensor)

print(f"{activations.keys() = }")
for key, val in activations.items():
    print(f"  {key}: {val.shape}")
