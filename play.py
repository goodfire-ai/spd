# %%
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

MODEL_SIZE = "1.25M"
model_path = f"SimpleStories/SimpleStories-{MODEL_SIZE}"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path)
model.to("cuda")
model.eval()

prompt = "map"

inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
input_ids = inputs.input_ids.to("cuda")

eos_token_id = 1

with torch.no_grad():
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=400,
        temperature=0.7,
        do_sample=True,
        eos_token_id=eos_token_id,
    )

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"\nGenerated text:\n{output_text}")

# %%
