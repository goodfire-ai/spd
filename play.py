# %%
import torch
from transformers import AutoTokenizer

from spd.models.component_model import ComponentModel


def get_spd_original_model_output(model, text, tokenizer):
    model.eval()

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids

    with torch.no_grad():
        # Get original model output
        output = model(input_ids, mode="target")

        print(output)


# Load tokenizer from your config
# model = ComponentModel.from_pretrained("model/model_200000.pth")
model = ComponentModel.from_pretrained("wandb:goodfire/spd-play/runs/og5fnp8l")

tokenizer = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-1.25M")

# Example usage with actual text
text = "Once upon a time there was a little girl named Lucy"

outputs = get_spd_original_model_output(model, text, tokenizer)
# %%
