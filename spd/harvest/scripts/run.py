"""Legacy entrypoint — delegates to run_worker and run_merge."""

import fire


def main(
    wandb_path: str,
    config_path: str | None = None,
    config_json: str | dict[str, object] | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    merge: bool = False,
    subrun_id: str | None = None,
) -> None:
    if merge:
        from spd.harvest.scripts.run_merge import main as merge_main

        assert subrun_id is not None, "--subrun_id required for --merge"
        merge_main(wandb_path, subrun_id)
    else:
        from spd.harvest.scripts.run_worker import main as worker_main

        worker_main(wandb_path, config_path, config_json, rank, world_size, subrun_id)


if __name__ == "__main__":
    fire.Fire(main)
