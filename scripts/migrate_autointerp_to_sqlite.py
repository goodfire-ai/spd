"""One-time migration: convert existing JSONL autointerp data to SQLite.

Usage:
    python scripts/migrate_autointerp_to_sqlite.py              # migrate all runs
    python scripts/migrate_autointerp_to_sqlite.py s-eab2ace8   # migrate specific run
"""

import json
import sys
from pathlib import Path

from spd.autointerp.db import InterpDB
from spd.autointerp.schemas import InterpretationResult
from spd.settings import SPD_OUT_DIR

AUTOINTERP_DIR = SPD_OUT_DIR / "autointerp"
HARVEST_DIR = SPD_OUT_DIR / "harvest"


# -- Old loader logic (copied from pre-migration loaders.py) --


def _load_results_from_jsonl(path: Path) -> dict[str, InterpretationResult]:
    results: dict[str, InterpretationResult] = {}
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            result = InterpretationResult(
                component_key=data["component_key"],
                label=data["label"],
                confidence=data["confidence"],
                reasoning=data.get("reasoning", ""),
                raw_response=data.get("raw_response", ""),
                prompt=data.get("prompt", ""),
            )
            results[result.component_key] = result
    return results


def _find_latest_nested_results(autointerp_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for subdir in autointerp_dir.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name in ("eval", "scoring"):
            continue
        results_path = subdir / "results.jsonl"
        if results_path.exists():
            candidates.append(results_path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.parent.name)
    return candidates[-1]


def _find_latest_flat_results(autointerp_dir: Path) -> Path | None:
    result_files = sorted(autointerp_dir.glob("results_*.jsonl"))
    if not result_files:
        # Also check for bare results.jsonl (written by on-demand endpoint)
        bare = autointerp_dir / "results.jsonl"
        if bare.exists():
            return bare
        return None
    return result_files[-1]


def _load_scores_from_jsonl(path: Path) -> list[tuple[str, float, str]]:
    """Load scores, returning (component_key, score, full_json_details) tuples."""
    scores: list[tuple[str, float, str]] = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            scores.append((data["component_key"], data["score"], line.strip()))
    return scores


def _find_latest_jsonl(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    candidates = sorted(directory.glob("results_*.jsonl"))
    if not candidates:
        return None
    return candidates[-1]


def _find_latest_autointerp_run_dir(autointerp_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for subdir in autointerp_dir.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name in ("eval", "scoring"):
            continue
        scoring_dir = subdir / "scoring"
        if scoring_dir.exists():
            candidates.append(subdir)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def migrate_run(run_id: str) -> bool:
    autointerp_dir = AUTOINTERP_DIR / run_id
    if not autointerp_dir.exists():
        print("  skip: no autointerp dir")
        return False

    db_path = autointerp_dir / "interp.db"
    if db_path.exists():
        print("  skip: interp.db already exists")
        return False

    db = InterpDB(db_path)
    migrated_anything = False

    # -- 1. Interpretations --
    # Try nested structure first
    nested = _find_latest_nested_results(autointerp_dir)
    if nested is not None:
        results = _load_results_from_jsonl(nested)
        if results:
            db.save_interpretations(list(results.values()))
            print(f"  interpretations: {len(results)} from {nested.relative_to(autointerp_dir)}")
            migrated_anything = True
    else:
        # Try flat structure
        flat = _find_latest_flat_results(autointerp_dir)
        if flat is not None:
            results = _load_results_from_jsonl(flat)
            if results:
                db.save_interpretations(list(results.values()))
                print(f"  interpretations: {len(results)} from {flat.relative_to(autointerp_dir)}")
                migrated_anything = True

    # -- 2. Intruder scores --
    # Check harvest dir first (current location)
    harvest_intruder_dir = HARVEST_DIR / run_id / "eval" / "intruder"
    intruder_path = _find_latest_jsonl(harvest_intruder_dir)

    # Also check legacy autointerp location
    if intruder_path is None:
        intruder_path = _find_latest_jsonl(autointerp_dir / "scoring" / "intruder")

    if intruder_path is not None:
        scores = _load_scores_from_jsonl(intruder_path)
        if scores:
            db.save_scores("intruder", scores)
            print(f"  intruder scores: {len(scores)} from {intruder_path}")
            migrated_anything = True

    # -- 3. Detection scores --
    run_dir = _find_latest_autointerp_run_dir(autointerp_dir)
    if run_dir is not None:
        detection_path = _find_latest_jsonl(run_dir / "scoring" / "detection")
        if detection_path is not None:
            scores = _load_scores_from_jsonl(detection_path)
            if scores:
                db.save_scores("detection", scores)
                print(
                    f"  detection scores: {len(scores)} "
                    f"from {detection_path.relative_to(autointerp_dir)}"
                )
                migrated_anything = True

        # -- 4. Fuzzing scores --
        fuzzing_path = _find_latest_jsonl(run_dir / "scoring" / "fuzzing")
        if fuzzing_path is not None:
            scores = _load_scores_from_jsonl(fuzzing_path)
            if scores:
                db.save_scores("fuzzing", scores)
                print(
                    f"  fuzzing scores: {len(scores)} "
                    f"from {fuzzing_path.relative_to(autointerp_dir)}"
                )
                migrated_anything = True

    db.close()

    if not migrated_anything:
        # Remove empty DB
        db_path.unlink()
        print("  skip: no data to migrate")
        return False

    return True


def main() -> None:
    if len(sys.argv) > 1:
        run_ids = sys.argv[1:]
    else:
        if not AUTOINTERP_DIR.exists():
            print(f"No autointerp dir at {AUTOINTERP_DIR}")
            return
        run_ids = sorted(d.name for d in AUTOINTERP_DIR.iterdir() if d.is_dir())

    migrated = 0
    for run_id in run_ids:
        print(f"{run_id}:")
        if migrate_run(run_id):
            migrated += 1

    print(f"\nDone: {migrated}/{len(run_ids)} runs migrated")


if __name__ == "__main__":
    main()
