"""CLI entry point for spd-pretrain command."""

import fire


def cli() -> None:
    from spd.pretrain.scripts.run_slurm import main

    fire.Fire(main)


if __name__ == "__main__":
    cli()
