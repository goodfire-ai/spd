"""Diff two memory-ledger roots (`memory_ledger` output): per shared program, buffer
bytes created / destroyed / resized between A and B, grouped by jax provenance — the
"what did my change do to memory" query. Buffer ids are compile-local, so the diff never
matches ids: it aggregates bytes-owning rows by provenance (op path, falling back to the
defining instruction name where jax recorded none) and compares the groups.

    python -m param_decomp.core.tools.ledger_diff <ledger_root_a> <ledger_root_b> \
        [--program <slug>] [--top N] [--peak-only]

`--peak-only` restricts to buffers live at each side's own arena peak (what moved the
high-water mark); default is every arena life (allocation churn anywhere in the step).
"""

import argparse
import dataclasses
from pathlib import Path

from param_decomp.core.tools.memory_ledger import (
    GIB,
    AllocationKind,
    ProgramLedger,
    read_program_ledger,
)


@dataclasses.dataclass(frozen=True)
class GroupStat:
    n_buffers: int
    total_bytes: int


@dataclasses.dataclass(frozen=True)
class GroupDelta:
    key: str
    a: GroupStat | None
    b: GroupStat | None

    @property
    def delta_bytes(self) -> int:
        return (self.b.total_bytes if self.b else 0) - (self.a.total_bytes if self.a else 0)


@dataclasses.dataclass(frozen=True)
class ProgramDiff:
    program: str
    peak_bytes: tuple[int, int]
    arena_bytes: tuple[int, int]
    parameter_bytes: tuple[int, int]
    created: tuple[GroupDelta, ...]
    destroyed: tuple[GroupDelta, ...]
    resized: tuple[GroupDelta, ...]


def provenance_groups(ledger: ProgramLedger, *, peak_only: bool) -> dict[str, GroupStat]:
    rows = ledger.live_at_peak() if peak_only else ledger.rows
    groups: dict[str, GroupStat] = {}
    for row in rows:
        if not (row.bytes_owner and row.allocation_kind is AllocationKind.ARENA):
            continue
        key = row.op_path or row.instruction
        prior = groups.get(key, GroupStat(0, 0))
        groups[key] = GroupStat(prior.n_buffers + 1, prior.total_bytes + row.size_bytes)
    return groups


def diff_programs(
    program: str, a: ProgramLedger, b: ProgramLedger, *, peak_only: bool
) -> ProgramDiff:
    groups_a = provenance_groups(a, peak_only=peak_only)
    groups_b = provenance_groups(b, peak_only=peak_only)
    created, destroyed, resized = [], [], []
    for key in sorted(groups_a.keys() | groups_b.keys()):
        delta = GroupDelta(key, groups_a.get(key), groups_b.get(key))
        match delta.a, delta.b:
            case None, _:
                created.append(delta)
            case _, None:
                destroyed.append(delta)
            case a_stat, b_stat if a_stat != b_stat:
                resized.append(delta)
            case _:
                pass
    by_magnitude = lambda d: -abs(d.delta_bytes)  # noqa: E731
    return ProgramDiff(
        program=program,
        peak_bytes=(a.peak_bytes, b.peak_bytes),
        arena_bytes=(a.arena_bytes, b.arena_bytes),
        parameter_bytes=(a.parameter_bytes, b.parameter_bytes),
        created=tuple(sorted(created, key=by_magnitude)),
        destroyed=tuple(sorted(destroyed, key=by_magnitude)),
        resized=tuple(sorted(resized, key=by_magnitude)),
    )


def _stat(stat: GroupStat | None) -> str:
    return "-" if stat is None else f"{stat.total_bytes / GIB:.3f} GiB x{stat.n_buffers}"


def render_diff(diff: ProgramDiff, top: int) -> str:
    def pair(label: str, values: tuple[int, int]) -> str:
        a_bytes, b_bytes = values
        return (
            f"{label} {a_bytes / GIB:.2f} -> {b_bytes / GIB:.2f} GiB"
            f" ({(b_bytes - a_bytes) / GIB:+.2f})"
        )

    lines = [
        f"== {diff.program}: {pair('peak', diff.peak_bytes)}; {pair('arena', diff.arena_bytes)};"
        f" {pair('parameters', diff.parameter_bytes)}"
    ]
    for label, deltas in [
        ("created (B only)", diff.created),
        ("destroyed (A only)", diff.destroyed),
        ("resized", diff.resized),
    ]:
        if not deltas:
            continue
        lines.append(f"  {label}: {len(deltas)} groups, showing top {min(top, len(deltas))}")
        for d in deltas[:top]:
            lines.append(
                f"    {d.delta_bytes / GIB:+9.3f} GiB  {_stat(d.a)} -> {_stat(d.b)}  {d.key}"
            )
    if not (diff.created or diff.destroyed or diff.resized):
        lines.append("  no provenance-group changes")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_a", type=Path)
    parser.add_argument("root_b", type=Path)
    parser.add_argument("--program", default=None, help="one program slug (default: all shared)")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--peak-only", action="store_true")
    args = parser.parse_args()

    def programs(root: Path) -> set[str]:
        found = {p.parent.relative_to(root).as_posix() for p in root.rglob("ledger.csv")}
        assert found, f"no ledger.csv under {root}"
        return found

    programs_a, programs_b = programs(args.root_a), programs(args.root_b)
    shared = programs_a & programs_b
    if args.program is not None:
        assert args.program in shared, (args.program, sorted(shared))
        shared = {args.program}
    for side, only in [("A", programs_a - programs_b), ("B", programs_b - programs_a)]:
        if only and args.program is None:
            print(f"programs only in {side}: {', '.join(sorted(only))}")
    for program in sorted(shared):
        diff = diff_programs(
            program,
            read_program_ledger(args.root_a / program),
            read_program_ledger(args.root_b / program),
            peak_only=args.peak_only,
        )
        print(render_diff(diff, args.top))
        print()


if __name__ == "__main__":
    main()
