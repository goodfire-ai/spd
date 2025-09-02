# Load a pretrained lm model
from spd.models.component_model import ComponentModel

model = ComponentModel.from_pretrained("wandb:goodfire/spd-play/runs/ggsc7d3q")
print(model)
print(model.state_dict().keys())
