"""Test script to reproduce caching behavior with different n_ctx values."""

from spd.data import create_data_loader_from_config
from spd.models.component_model import SPDRunInfo

WANDB_PATH = "wandb:goodfire/spd/runs/jyo9duz5"


def main():
    print("Loading run info...")
    run_info = SPDRunInfo.from_path(WANDB_PATH)
    config = run_info.config

    print("\n" + "=" * 60)
    print("Creating dataloader with n_ctx=8 (first time)")
    print("=" * 60)
    loader1, _ = create_data_loader_from_config(config, batch_size=4, n_ctx=8)

    print("\n" + "=" * 60)
    print("Creating dataloader with n_ctx=16")
    print("=" * 60)
    loader2, _ = create_data_loader_from_config(config, batch_size=4, n_ctx=16)

    print("\n" + "=" * 60)
    print("Creating dataloader with n_ctx=8 (second time - should be cached)")
    print("=" * 60)
    loader3, _ = create_data_loader_from_config(config, batch_size=4, n_ctx=8)

    print("\nDone! Check if the third call was cached or re-processed.")


if __name__ == "__main__":
    main()
