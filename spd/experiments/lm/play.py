from __future__ import annotations

# %%
# Example / sandbox script for running ComponentModel on a pretrained model.

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, LinearComponent

# %%
print("Loading base language model ...")

model_path = "SimpleStories/SimpleStories-1.25M"
assert model_path is not None, (
    "`pretrained_model_path` must be specified in the config when using ComponentModel."
)

base_model = LlamaForCausalLM.from_pretrained(model_path)

# %%
# Select the model size you want to use
model_path = "SimpleStories/SimpleStories-1.25M"

# Load the base model
model = LlamaForCausalLM.from_pretrained(model_path, device_map="cuda")
# model.to("cuda")

# %%

# ------------------------------------------------------------------
# Build ComponentModel
# ------------------------------------------------------------------
comp_model = ComponentModel(
    base_model=model,
    target_module_patterns=["model.model.layers.*.mlp.gate_proj"],
    C=17,
    n_ci_mlp_neurons=0,
    pretrained_model_output_attr="logits",
)

# # Create components with rank=10 (adjust as needed)
# gate_proj_components = create_target_components(
#     model, rank=C, target_module_patterns=["model.transformer.h.*.mlp.gate_proj"]
# )
gate_proj_components: dict[str, LinearComponent | EmbeddingComponent] = {
    k.removeprefix("components.").replace("-", "."): v for k, v in comp_model.components.items()
}  # type: ignore
# %%
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)

# Define your prompt
prompt = "The curious cat looked at the"

# IMPORTANT: Use tokenizer without special tokens
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
# input_ids = inputs.input_ids.to("cuda")
input_ids = inputs.input_ids.to("cuda")
# Targets should be the inputs shifted by one (we will later ignore the last input token)
targets = input_ids[:, 1:]
input_ids = input_ids[:, :-1]

# IMPORTANT: Set correct EOS token ID (not the default from tokenizer)
eos_token_id = 1

# %%

# # Generate text
# with torch.no_grad():
#     output_ids = model.generate(
#         idx=input_ids, max_new_tokens=20, temperature=0.7, top_k=40, eos_token_id=eos_token_id
#     )

# # Decode output
# output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
# print(f"Generated text:\n{output_text}")


# %%

# logits, _ = ss_model.forward(input_ids, components=gate_proj_components)
logits = comp_model.forward(input_ids).logits
print("inputs_shape", input_ids.shape)
print("logits", logits)
print("logits shape", logits.shape)

logits = comp_model.forward_with_components(input_ids, components=gate_proj_components)

print("Component logits shape", logits.shape)
print("Component logits", logits)

# Create some dummy masks
masks = {
    f"model.model.layers.{i}.mlp.gate_proj": torch.randn(1, input_ids.shape[-1], comp_model.C)
    for i in range(len(model.model.layers))
}

logits = comp_model.forward_with_components(input_ids, components=gate_proj_components, masks=masks)

print("Masked component logits shape", logits.shape)
print("Masked component logits", logits)
#########################################################
# %%
