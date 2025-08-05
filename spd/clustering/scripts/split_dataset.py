import json
from pathlib import Path

import numpy as np
import torch
from muutils.spinner import SpinnerContext
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.models.component_model import SPDRunInfo


def split_dataset(
    model_path: str,
    n_batches: int,
    batch_size: int,
    base_path: Path = Path("data/split_datasets"),
    save_file_fmt: str = "batchsize_{batch_size}/batch_{batch_idx}.npz",
    cfg_file_fmt: str = "batchsize_{batch_size}/_config.json",
) -> Path:
    """split up a SS dataset into n_batches of batch_size, returned the saved paths

    1. load the config for a SimpleStories SPD Run given by model_path
    2. create a DataLoader for the dataset
    3. iterate over the DataLoader and save each batch to a file


    """
    with SpinnerContext(message="Loading SPD Run Config..."):
        spd_run: SPDRunInfo = SPDRunInfo.from_path(model_path)
        cfg: Config = spd_run.config

    dataset_config: DatasetConfig = DatasetConfig(
        name=cfg.task_config.dataset_name,
        hf_tokenizer_path=cfg.pretrained_model_name_hf,
        split=cfg.task_config.train_data_split,
        n_ctx=cfg.task_config.max_seq_len,
        is_tokenized=False,
        streaming=False,
        seed=0,
        column_name=cfg.task_config.column_name,
    )

    with SpinnerContext(message="getting dataloader..."):
        dataloader: DataLoader[dict[str, torch.Tensor]]
        dataloader, _tokenizer = create_data_loader(
            dataset_config=dataset_config,
            batch_size=batch_size,
            buffer_size=cfg.task_config.buffer_size,
            global_seed=cfg.seed,
            ddp_rank=0,
            ddp_world_size=1,
        )

    # make dirs
    base_path.mkdir(parents=True, exist_ok=True)
    (
        base_path
        / save_file_fmt.format(batch_size=batch_size, batch_idx="XX", n_batches=f"{n_batches:02d}")
    ).parent.mkdir(parents=True, exist_ok=True)
    # iterate over the requested number of batches and save them
    output_paths: list[Path] = []
    for batch_idx, batch in tqdm(
        enumerate(iter(dataloader)),
        total=n_batches,
        unit="batche",
    ):
        if batch_idx >= n_batches:
            break
        batch_path: Path = base_path / save_file_fmt.format(
            batch_size=batch_size,
            batch_idx=f"{batch_idx:02d}",
            n_batches=f"{n_batches:02d}",
        )
        np.savez_compressed(
            batch_path,
            input_ids=batch["input_ids"].cpu().numpy(),
        )
        output_paths.append(batch_path)

    # save a config file
    cfg_path: Path = base_path / cfg_file_fmt.format(batch_size=batch_size)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        json.dumps(
            dict(
                # args to this function
                model_path=model_path,
                batch_size=batch_size,
                n_batches=n_batches,
                # dataset and tokenizer config
                dataset_config=dataset_config.model_dump(mode="json"),
                tokenizer_path=str(getattr(_tokenizer, "name_or_path", None)),
                tokenizer_type=str(getattr(_tokenizer, "__class__", None)),
                # files we saved
                output_files=[str(p) for p in output_paths],
                output_dir=str(base_path),
                output_file_fmt=save_file_fmt,
                cfg_file_fmt=cfg_file_fmt,
                cfg_file=str(cfg_path),
            ),
            indent="\t",
        )
    )

    print(f"Saved config to: {cfg_path}")

    return cfg_path


if __name__ == "__main__":
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="tokenize and split a SimpleStories dataset into smaller batches for clustering ensemble",
    )

    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        default="wandb:goodfire/spd/runs/ioprgffh",
        help="Path to the SPD Run, usually a wandb run",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        required=True,
        help="Batch size for the DataLoader, each batch will be saved to a separate file",
    )
    parser.add_argument(
        "--n-batches",
        "-n",
        type=int,
        required=True,
        help="Number of batches to split the dataset into",
    )
    parser.add_argument(
        "--base-path",
        "-p",
        type=Path,
        default=Path("data/split_datasets"),
        help="Base path to save the split datasets to",
    )

    args: argparse.Namespace = parser.parse_args()

    split_dataset(
        model_path=args.model_path,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        base_path=args.base_path,
    )
