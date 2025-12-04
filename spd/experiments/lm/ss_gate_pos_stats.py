"""Compute SimpleStories gate activation statistics.

This script loads the SimpleStories 1.25M parameter model, tokenizes
ten training examples to sequence length 512, and performs a forward
pass. It records, for each MLP gate projection, the average number of
outputs that are (a) positive and (b) positive with magnitude greater
than 0.1, averaged across all sequences and positions.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "SimpleStories/SimpleStories-1.25M"
DATASET_NAME = "SimpleStories/SimpleStories"
DATASET_SPLIT = "train"
DATASET_TEXT_COLUMN = "story"
SEQ_LEN = 512
BATCH_SIZE = 10
MAG_THRESHOLD = 1.0


def _select_batch(tokenizer: AutoTokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    """Return input ids and attention mask for a batch of sequences of length 512."""

    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    input_ids_list: list[torch.Tensor] = []

    for example in dataset:
        text = example[DATASET_TEXT_COLUMN]
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=SEQ_LEN,
            padding=False,
            add_special_tokens=False,
        )
        length = encoded["input_ids"].shape[-1]
        if length >= SEQ_LEN:
            tokens = encoded["input_ids"][..., :SEQ_LEN].contiguous()
            input_ids_list.append(tokens)
        if len(input_ids_list) == BATCH_SIZE:
            break

    if len(input_ids_list) != BATCH_SIZE:
        raise RuntimeError(
            f"Failed to find {BATCH_SIZE} examples with at least {SEQ_LEN} tokens in "
            f"{DATASET_NAME}/{DATASET_SPLIT}."
        )

    input_ids = torch.cat(input_ids_list, dim=0)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids, attention_mask = _select_batch(tokenizer)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    gate_outputs: OrderedDict[int, torch.Tensor] = OrderedDict()
    hooks = []

    def _make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            gate_outputs[layer_idx] = output.detach().to("cpu")

        return hook

    for idx, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.gate_proj.register_forward_hook(_make_hook(idx)))

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    for hook in hooks:
        hook.remove()

    if len(gate_outputs) == 0:
        raise RuntimeError("No gate activations were captured.")

    for layer_idx, activations in gate_outputs.items():
        if activations.dim() == 2:
            activations = activations.unsqueeze(0)
        positives_per_position = (activations > 0).sum(dim=-1).float()
        positive_and_large_mask = (activations > 0) & (activations.abs() > MAG_THRESHOLD)
        positives_large_per_position = positive_and_large_mask.sum(dim=-1).float()
        avg_positive_scalar = positives_per_position.mean().item()
        avg_positive_large_scalar = positives_large_per_position.mean().item()
        total_units = activations.shape[-1]
        print(
            f"Layer {layer_idx}: average positive gate activations per position = "
            f"{avg_positive_scalar:.2f} / {total_units}"
        )
        print(
            f"Layer {layer_idx}: average positive gate activations > {MAG_THRESHOLD} = "
            f"{avg_positive_large_scalar:.2f} / {total_units}"
        )


if __name__ == "__main__":
    main()
