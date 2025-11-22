# %%
from typing import Any

import torch
from simple_stories_train.run_info import RunInfo as SSRunInfo
from transformers import AutoTokenizer, LlamaForCausalLM

from spd.utils.general_utils import resolve_class

# %%
models_to_test: list[dict[str, str]] = []

# Existing HF models
for MODEL_SIZE in ["1.25M", "5M", "11M", "30M", "35M"]:
    models_to_test.append(
        {
            "type": "hf",
            "name": f"SimpleStories/SimpleStories-V2-{MODEL_SIZE}",
            "path": f"SimpleStories/SimpleStories-V2-{MODEL_SIZE}",
        }
    )

# New LlamaSimple models
models_to_test.append(
    {
        "type": "ls",
        "name": "LlamaSimple-1L (tfacbi70)",
        "id": "tfacbi70",
        "tokenizer": "SimpleStories/test-SimpleStories-gpt2-1.25M",
    }
)
models_to_test.append(
    {
        "type": "ls",
        "name": "LlamaSimple-2L (tb8373uo)",
        "id": "tb8373uo",
        "tokenizer": "SimpleStories/test-SimpleStories-gpt2-1.25M",
    }
)


def load_model(model_info: dict[str, str]) -> tuple[Any, Any]:
    print(f"Loading {model_info['name']}...")
    if model_info["type"] == "hf":
        model = LlamaForCausalLM.from_pretrained(model_info["path"])
        tokenizer = AutoTokenizer.from_pretrained(model_info["path"])
    elif model_info["type"] == "ls":
        model_class = resolve_class("simple_stories_train.models.llama_simple.LlamaSimple")
        run_path = f"wandb:goodfire/spd/runs/{model_info['id']}"
        # We need to handle authentication if needed, but assuming env is set up
        run_info = SSRunInfo.from_path(run_path)
        model = model_class.from_run_info(run_info)  # pyright: ignore[reportAttributeAccessIssue]
        tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer"])
    else:
        raise ValueError(f"Unknown model type: {model_info['type']}")

    return model, tokenizer


for model_info in models_to_test:
    model, tokenizer = load_model(model_info)
    model.to("cuda")
    model.eval()

    # prompt = "Tom lost his favorite toy, which made him very"
    # prompt = '"What do you mean?'
    prompt = "They walked hand in"
    # prompt = "When Alex and Leo went to the store, Alex gave a drink to"

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids.to("cuda")

    eos_token_id = 1

    print(f"Model: {model_info['name']}")
    print(f"Prompt: {prompt}")
    print("Generating text...")
    for _ in range(10):
        with torch.no_grad():
            if model_info["type"] == "hf":
                output_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=5,
                    temperature=0.7,
                    do_sample=True,
                    eos_token_id=eos_token_id,
                    pad_token_id=eos_token_id,
                    attention_mask=torch.ones_like(input_ids),
                )
            else:
                output_ids = model.generate(
                    idx=input_ids,
                    max_new_tokens=5,
                    temperature=0.7,
                    eos_token_id=eos_token_id,
                )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(output_text)

    print("-" * 100)

# %%
