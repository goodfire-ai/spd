"""Convert retired warmup/decay schedules to exact knot schedules.

The retired shape was::

    {start_val, warmup_pct, final_val_frac, fn_type}

The replacement is ``{max_val, points}``. ``migrate_raw`` walks one complete
experiment config and converts every retired schedule in memory. The LM loader uses
it at its stored-config boundary, so immutable pre-knot ``launch_config.yaml`` files can
resume without being rewritten; newly-authored configs still validate only against the
current schema.

To convert a standalone config without mutating the source::

    python -m param_decomp.migrations.schedule_knots OLD.yaml > NEW.yaml
"""

import copy
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

Interp = Literal["constant", "cosine", "linear"]
_LEGACY_KEYS = frozenset({"start_val", "warmup_pct", "final_val_frac", "fn_type"})


def _point(at: float, frac: float, interp: str = "linear") -> dict[str, float | str]:
    return {"at": at, "frac": frac, "interp": interp}


def migrate_schedule(raw: dict[str, Any], total_steps: int) -> dict[str, Any]:
    """Convert one retired schedule while preserving every trained-step value exactly.

    The old warmup crossed at the integer ``int(total_steps * warmup_pct)``; placing the
    knot at that step's normalized time preserves the discrete trajectory, unlike using
    ``warmup_pct`` directly. ``max_val`` is rescaled when the retired final fraction is
    above one, keeping the new schema's promise that a knot actually attains the peak.
    """
    assert "start_val" in raw and set(raw) <= _LEGACY_KEYS, raw
    assert total_steps > 0, total_steps

    start = float(raw["start_val"])
    warmup_pct = float(raw.get("warmup_pct", 0.0))
    final_frac = float(raw.get("final_val_frac", 1.0))
    fn_type: Interp = raw.get("fn_type", "constant")
    assert start > 0.0 and 0.0 <= warmup_pct <= 1.0 and final_frac >= 0.0, raw
    assert fn_type in ("constant", "cosine", "linear"), raw
    if fn_type == "constant":
        assert final_frac == 1.0, raw

    warmup_steps = int(total_steps * warmup_pct)
    decay_steps = total_steps - warmup_steps

    # With one trained step, the old evaluator returned zero only for a full warmup;
    # otherwise its decay_steps<=1 arm returned start without consulting fn_type.
    if total_steps == 1:
        first_frac = 0.0 if warmup_steps == 1 else 1.0
        return {
            "max_val": start,
            "points": [_point(0.0, first_frac), _point(1.0, 1.0)],
        }

    # A 100% warmup never reaches start on a trained step. Rescaling the peak makes
    # the normalized 0->1 line reproduce start * step / total_steps pointwise.
    if warmup_steps == total_steps:
        trained_peak = start * (total_steps - 1) / total_steps
        return {
            "max_val": trained_peak,
            "points": [_point(0.0, 0.0), _point(1.0, 1.0)],
        }

    # Exactly one decay step takes the retired evaluator's start-valued short arm.
    if decay_steps == 1:
        return {
            "max_val": start,
            "points": [_point(0.0, 0.0), _point(1.0, 1.0)],
        }

    end = start if fn_type == "constant" else start * final_frac
    peak = max(start, end)
    start_frac = start / peak
    end_frac = end / peak
    end_interp = "linear" if fn_type == "constant" else fn_type

    if warmup_steps == 0:
        points = [_point(0.0, start_frac), _point(1.0, end_frac, end_interp)]
    else:
        crossing = warmup_steps / (total_steps - 1)
        points = [
            _point(0.0, 0.0),
            _point(crossing, start_frac),
            _point(1.0, end_frac, end_interp),
        ]
    return {"max_val": peak, "points": points}


def _migrate_node(node: Any, total_steps: int) -> Any:
    if isinstance(node, dict):
        if "start_val" in node and set(node) <= _LEGACY_KEYS:
            return migrate_schedule(node, total_steps)
        return {key: _migrate_node(value, total_steps) for key, value in node.items()}
    if isinstance(node, list):
        return [_migrate_node(value, total_steps) for value in node]
    return copy.deepcopy(node)


def migrate_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a current-schema copy of one raw experiment config.

    Current configs are returned unchanged (apart from the defensive deep copy). A raw
    config containing a retired schedule must carry ``pd.steps`` because the exact knot
    crossing depends on the run's discrete step count.
    """
    if not contains_retired_schedule(raw):
        return copy.deepcopy(raw)
    total_steps = raw.get("pd", {}).get("steps")
    assert isinstance(total_steps, int) and total_steps > 0, (
        "a config with retired schedules needs positive integer pd.steps for exact migration"
    )
    return _migrate_node(raw, total_steps)


def contains_retired_schedule(node: Any) -> bool:
    if isinstance(node, dict):
        if "start_val" in node and set(node) <= _LEGACY_KEYS:
            return True
        return any(contains_retired_schedule(value) for value in node.values())
    if isinstance(node, list):
        return any(contains_retired_schedule(value) for value in node)
    return False


def retired_value(step: int, total_steps: int, raw: dict[str, Any]) -> float:
    """Executable reference for the pre-knot evaluator."""
    start = float(raw["start_val"])
    warmup_pct = float(raw.get("warmup_pct", 0.0))
    final_frac = float(raw.get("final_val_frac", 1.0))
    fn_type = raw.get("fn_type", "constant")
    warmup_steps = int(total_steps * warmup_pct)
    decay_steps = total_steps - warmup_steps
    if step < warmup_steps:
        return start * step / warmup_steps
    if decay_steps <= 1:
        return start
    progress = (step - warmup_steps) / (decay_steps - 1)
    match fn_type:
        case "constant":
            return start
        case "linear":
            return start * (final_frac + (1.0 - final_frac) * (1.0 - progress))
        case "cosine":
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return start * (final_frac + (1.0 - final_frac) * cosine)
        case _:
            raise AssertionError(fn_type)


def main(argv: list[str]) -> int:
    assert len(argv) == 1, __doc__
    source = Path(argv[0])
    raw = yaml.safe_load(source.read_text())
    assert isinstance(raw, dict), source
    sys.stdout.write(yaml.safe_dump(migrate_raw(raw), sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
