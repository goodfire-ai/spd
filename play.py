# %%
import os

from spd.models.component_model import ComponentModel

wandb_api_key = os.getenv("WANDB_API_KEY")


def load_tms_5_2() -> ComponentModel:
    """
    Load the canonical TMS 5-2 run from Weights & Biases.

    Returns
    -------
    ComponentModel
        The loaded component model.
    """
    canonical_run = "wandb:goodfire/spd/9igruzrm"
    # Downloads the checkpoint (or uses a cached copy) and builds the model.
    model: ComponentModel = ComponentModel.from_pretrained(canonical_run)
    return model


if __name__ == "__main__":
    model = load_tms_5_2()
    print("Loaded `tms_5-2` ComponentModel successfully!")
# ... existing code ...
