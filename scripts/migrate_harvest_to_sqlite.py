"""One-time migration: convert existing JSONL harvest data to SQLite.

Usage:
    python scripts/migrate_harvest_to_sqlite.py              # migrate all runs
    python scripts/migrate_harvest_to_sqlite.py s-eab2ace8   # migrate specific run
"""

import json
import sys

from spd.harvest.config import HarvestConfig
from spd.harvest.db import HarvestDB
from spd.harvest.schemas import (
    ActivationExample,
    ComponentData,
    ComponentTokenPMI,
    get_activation_contexts_dir,
)
from spd.settings import SPD_OUT_DIR

HARVEST_DIR = SPD_OUT_DIR / "harvest"


def migrate_run(run_id: str) -> bool:
    ac_dir = get_activation_contexts_dir(run_id)
    components_path = ac_dir / "components.jsonl"
    config_path = ac_dir / "config.json"
    db_path = ac_dir / "harvest.db"

    if not components_path.exists():
        print("  skip: no components.jsonl")
        return False

    if db_path.exists():
        print("  skip: harvest.db already exists")
        return False

    # Read config
    config = None
    if config_path.exists():
        try:
            with open(config_path) as f:
                raw = json.load(f)
            valid_fields = set(HarvestConfig.model_fields.keys())
            config = HarvestConfig(**{k: v for k, v in raw.items() if k in valid_fields})
        except Exception as e:
            print(f"  warning: config parse failed ({e}), skipping config")

    # Read components
    components: list[ComponentData] = []
    with open(components_path) as f:
        for line in f:
            data = json.loads(line)
            examples = [
                ActivationExample(
                    token_ids=ex["token_ids"],
                    ci_values=ex["ci_values"],
                    component_acts=ex.get("component_acts", [0.0] * len(ex["ci_values"])),
                )
                for ex in data["activation_examples"]
            ]
            components.append(
                ComponentData(
                    component_key=data["component_key"],
                    layer=data["layer"],
                    component_idx=data["component_idx"],
                    mean_ci=data["mean_ci"],
                    activation_examples=examples,
                    input_token_pmi=ComponentTokenPMI(**data["input_token_pmi"]),
                    output_token_pmi=ComponentTokenPMI(**data["output_token_pmi"]),
                )
            )

    # Write to SQLite
    db = HarvestDB(db_path)
    db.save_components(components)
    if config is not None:
        db.save_config(config)
    db.close()

    print(f"  migrated: {len(components)} components -> harvest.db")
    return True


def main() -> None:
    if len(sys.argv) > 1:
        run_ids = sys.argv[1:]
    else:
        run_ids = sorted(d.name for d in HARVEST_DIR.iterdir() if d.is_dir())

    migrated = 0
    for run_id in run_ids:
        print(f"{run_id}:")
        if migrate_run(run_id):
            migrated += 1

    print(f"\nDone: {migrated}/{len(run_ids)} runs migrated")


if __name__ == "__main__":
    main()
