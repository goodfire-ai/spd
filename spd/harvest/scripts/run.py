"""Harvest worker entrypoint. See run_worker.py."""

from spd.harvest.scripts.run_worker import main

if __name__ == "__main__":
    import fire

    fire.Fire(main)
