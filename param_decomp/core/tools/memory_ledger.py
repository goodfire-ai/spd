"""Per-program memory ledger from an XLA buffer-assignment dump: every assigned buffer
as one typed row — size, allocation (arena/parameter/static) + offset, heap live range,
defining HLO instruction, and jax provenance (op path + source file:line) — plus the
derived peak snapshot ("what is co-resident at the arena's high-water instruction").

Data source: the `*after_optimizations.hlo.pb` a compile writes under
`xla_dump_to=<dir>` with `xla_dump_hlo_as_proto=true` — the `HloProto` carrying the
optimized module AND its `BufferAssignmentProto` (decoded by `memreport`'s fail-closed
parsers). The deviceless fit check emits these per program via `--ledger <dir>`
(`python -m param_decomp.experiments.lm.fit_check`); any dump directory works:

    python -m param_decomp.core.tools.memory_ledger <dump_dir> --out <ledger_root>

Per program the ledger dir gets `ledger.csv` (the greppable table), `meta.json`
(module-level totals), and `peak_report.txt` (the full live-at-peak list, size-sorted).
`ledger_diff` compares two such roots. The same `.hlo.pb` is what the profiler's
memory viewer consumes for its buffer-assignment views — keep it (the fit check leaves
it under `<program>/dump/`).

Row semantics: rows with `bytes_owner` own their bytes exactly once — arena rows are the
heap-simulator sweep's canonical live ranges (a buffer the collective-pipeliner reallocs
appears once per life), non-arena rows are one representative per assigned
(allocation, offset) slice; the remaining rows are XLA's aliases of those slices
(bitcast/tuple views: no bytes, no live range). Sum `size_bytes` over `bytes_owner`
rows live at `peak_event` and you reproduce `peak_bytes` — asserted at build time.
"""

import argparse
import csv
import dataclasses
import json
from collections import defaultdict
from enum import Enum
from pathlib import Path

from param_decomp.core.tools.memreport import (
    BufferAssignment,
    Instruction,
    StackFrameIndex,
    _render_shape,
    parse_buffer_assignment,
    parse_instruction_index,
    parse_module_name,
    parse_stack_frame_index,
    sweep_live_ranges,
)

GIB = 2**30


class AllocationKind(Enum):
    ARENA = "arena"
    PARAMETER = "parameter"
    STATIC = "static"


@dataclasses.dataclass(frozen=True)
class LedgerRow:
    """One assigned buffer (or one arena life of one). `born_event`/`dies_event` index
    the whole-module heap-simulator trace — program order; live over [born, dies).
    They are None off the arena: parameters/statics are resident for the whole program."""

    buffer_id: int
    size_bytes: int
    allocation_index: int
    allocation_kind: AllocationKind
    offset: int
    bytes_owner: bool
    born_event: int | None
    dies_event: int | None
    instruction: str
    opcode: str
    shape: str
    op_path: str
    source: str


@dataclasses.dataclass(frozen=True)
class ProgramLedger:
    module_name: str
    peak_bytes: int
    peak_event: int
    peak_instruction: str
    n_events: int
    arena_bytes: int
    parameter_bytes: int
    static_bytes: int
    rows: tuple[LedgerRow, ...]

    def live_at_peak(self) -> tuple[LedgerRow, ...]:
        live = tuple(
            row
            for row in self.rows
            if row.bytes_owner
            and row.born_event is not None
            and row.dies_event is not None
            and row.born_event <= self.peak_event < row.dies_event
        )
        return tuple(sorted(live, key=lambda r: -r.size_bytes))


def _row(
    assignment: BufferAssignment,
    instructions: dict[int, Instruction],
    frames: StackFrameIndex,
    buffer_id: int,
    kind: AllocationKind,
    *,
    bytes_owner: bool,
    born_event: int | None,
    dies_event: int | None,
) -> LedgerRow:
    buffer = assignment.buffers[buffer_id]
    location = assignment.buffer_locations[buffer_id]
    instruction = instructions[buffer.defining_instruction_id]
    return LedgerRow(
        buffer_id=buffer_id,
        size_bytes=buffer.size_bytes,
        allocation_index=location.allocation_index,
        allocation_kind=kind,
        offset=location.offset,
        bytes_owner=bytes_owner,
        born_event=born_event,
        dies_event=dies_event,
        instruction=instruction.name,
        opcode=instruction.opcode,
        shape=_render_shape(instruction.shape, buffer.shape_index),
        op_path=instruction.op_path,
        source=frames.source_of(instruction.stack_frame_id),
    )


def build_program_ledger(hlo_proto: memoryview) -> ProgramLedger:
    """Pure derivation from one dumped `HloProto`; every accounting identity the ledger
    claims is asserted here, so a ledger that writes is a ledger that adds up."""
    assignment = parse_buffer_assignment(hlo_proto)
    instructions = parse_instruction_index(hlo_proto)
    frames = parse_stack_frame_index(hlo_proto)
    liveness = sweep_live_ranges(assignment)

    kind_of_allocation = {
        i: (
            AllocationKind.ARENA
            if i == assignment.arena_allocation_index
            else AllocationKind.PARAMETER
            if i in assignment.entry_parameter_allocation_indices
            else AllocationKind.STATIC
        )
        for i in range(len(assignment.allocation_sizes))
    }

    rows: list[LedgerRow] = []
    arena_owner_ids = set()
    for live_range in liveness.ranges:
        arena_owner_ids.add(live_range.buffer_id)
        rows.append(
            _row(
                assignment,
                instructions,
                frames,
                live_range.buffer_id,
                AllocationKind.ARENA,
                bytes_owner=True,
                born_event=live_range.born,
                dies_event=live_range.dies,
            )
        )

    slices: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for buffer_id, location in assignment.buffer_locations.items():
        match kind_of_allocation[location.allocation_index]:
            case AllocationKind.ARENA:
                if buffer_id not in arena_owner_ids:
                    rows.append(
                        _row(
                            assignment,
                            instructions,
                            frames,
                            buffer_id,
                            AllocationKind.ARENA,
                            bytes_owner=False,
                            born_event=None,
                            dies_event=None,
                        )
                    )
            case AllocationKind.PARAMETER | AllocationKind.STATIC:
                slices[(location.allocation_index, location.offset)].append(buffer_id)
    for slice_buffer_ids in slices.values():
        owner_id, *alias_ids = sorted(
            slice_buffer_ids, key=lambda b: (-assignment.buffers[b].size_bytes, b)
        )
        for buffer_id in [owner_id, *alias_ids]:
            rows.append(
                _row(
                    assignment,
                    instructions,
                    frames,
                    buffer_id,
                    kind_of_allocation[assignment.buffer_locations[buffer_id].allocation_index],
                    bytes_owner=buffer_id == owner_id,
                    born_event=None,
                    dies_event=None,
                )
            )

    arena_bytes = assignment.allocation_sizes[assignment.arena_allocation_index]
    parameter_bytes = sum(
        assignment.allocation_sizes[i] for i in assignment.entry_parameter_allocation_indices
    )
    static_bytes = sum(assignment.allocation_sizes) - arena_bytes - parameter_bytes
    ledger = ProgramLedger(
        module_name=parse_module_name(hlo_proto),
        peak_bytes=liveness.peak_bytes,
        peak_event=liveness.peak_event,
        peak_instruction=assignment.events[liveness.peak_event].instruction_name,
        n_events=liveness.n_events,
        arena_bytes=arena_bytes,
        parameter_bytes=parameter_bytes,
        static_bytes=static_bytes,
        rows=tuple(rows),
    )

    assert ledger.peak_bytes <= arena_bytes, "swept peak exceeds the arena allocation"
    at_peak = ledger.live_at_peak()
    assert sum(r.size_bytes for r in at_peak) == ledger.peak_bytes, (
        "live-at-peak rows do not sum to the swept peak"
    )
    intervals = sorted((r.offset, r.size_bytes) for r in at_peak)
    for (offset, size), (next_offset, _) in zip(intervals, intervals[1:], strict=False):
        assert offset + size <= next_offset, "peak-live buffers overlap in the arena"
    return ledger


# ── I/O: one directory per program ───────────────────────────────────────────────────

_CSV_COLUMNS = [f.name for f in dataclasses.fields(LedgerRow)]


def render_peak_report(ledger: ProgramLedger, top: int | None) -> str:
    """The peak snapshot: the high-water program point and every buffer live there,
    size-sorted with provenance (`top=None` = all of them)."""
    at_peak = ledger.live_at_peak()
    lines = [
        f"module: {ledger.module_name}",
        f"arena peak: {ledger.peak_bytes / GIB:.2f} GiB of {ledger.arena_bytes / GIB:.2f} GiB"
        f" arena (fragmentation {(ledger.arena_bytes - ledger.peak_bytes) / GIB:.2f})"
        f" at event {ledger.peak_event}/{ledger.n_events} ({ledger.peak_instruction})",
        f"resident outside the arena: parameters {ledger.parameter_bytes / GIB:.2f} GiB"
        f" + static {ledger.static_bytes / GIB:.2f} GiB",
        f"live at peak: {len(at_peak)} buffers"
        + (f" (top {top} below)" if top is not None and top < len(at_peak) else ""),
        "",
        "    size  [born, dies)         shape  instruction | op_path | source",
    ]
    for r in at_peak[: len(at_peak) if top is None else top]:
        lines.append(
            f"{r.size_bytes / GIB:8.3f}  [{r.born_event:>7}, {r.dies_event:>7})"
            f"  {r.shape:>24}  {r.instruction} | {r.op_path} | {r.source}"
        )
    return "\n".join(lines)


def write_program_ledger(ledger: ProgramLedger, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "ledger.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_COLUMNS)
        for row in ledger.rows:
            writer.writerow(
                [
                    row.buffer_id,
                    row.size_bytes,
                    row.allocation_index,
                    row.allocation_kind.value,
                    row.offset,
                    str(row.bytes_owner).lower(),
                    "" if row.born_event is None else row.born_event,
                    "" if row.dies_event is None else row.dies_event,
                    row.instruction,
                    row.opcode,
                    row.shape,
                    row.op_path,
                    row.source,
                ]
            )
    meta = {
        f.name: getattr(ledger, f.name)
        for f in dataclasses.fields(ProgramLedger)
        if f.name != "rows"
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out_dir / "peak_report.txt").write_text(render_peak_report(ledger, top=None) + "\n")


def read_program_ledger(ledger_dir: Path) -> ProgramLedger:
    meta = json.loads((ledger_dir / "meta.json").read_text())
    with (ledger_dir / "ledger.csv").open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == _CSV_COLUMNS, (ledger_dir, header)
        rows = tuple(
            LedgerRow(
                buffer_id=int(r[0]),
                size_bytes=int(r[1]),
                allocation_index=int(r[2]),
                allocation_kind=AllocationKind(r[3]),
                offset=int(r[4]),
                bytes_owner={"true": True, "false": False}[r[5]],
                born_event=None if r[6] == "" else int(r[6]),
                dies_event=None if r[7] == "" else int(r[7]),
                instruction=r[8],
                opcode=r[9],
                shape=r[10],
                op_path=r[11],
                source=r[12],
            )
            for r in reader
        )
    return ProgramLedger(rows=rows, **meta)


def emit_program_ledger(dump_dir: Path, out_dir: Path) -> ProgramLedger:
    """Build and write the ledger for the ONE program dumped under `dump_dir`."""
    (hlo_pb,) = sorted(dump_dir.glob("*after_optimizations.hlo.pb"))
    ledger = build_program_ledger(memoryview(hlo_pb.read_bytes()))
    write_program_ledger(ledger, out_dir)
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", type=Path, help="an xla_dump_to dir (protos enabled)")
    parser.add_argument("--out", type=Path, required=True, help="ledger root (one subdir/program)")
    parser.add_argument("--top", type=int, default=20, help="peak-report rows to print")
    args = parser.parse_args()

    hlo_pbs = sorted(args.dump_dir.rglob("*after_optimizations.hlo.pb"))
    assert hlo_pbs, (
        f"no *after_optimizations.hlo.pb under {args.dump_dir} — dump with"
        " xla_dump_hlo_as_proto=true (the fit check's --ledger does)"
    )
    # module_<id>.<name>.<pipeline>_after_optimizations.hlo.pb — slug by name, module id
    # appended only to break ties (several jit_eval_step programs in one dump dir).
    names = [p.name.split(".")[1] for p in hlo_pbs]
    for path, name in zip(hlo_pbs, names, strict=True):
        module_id = path.name.split(".")[0].removeprefix("module_")
        slug = name if names.count(name) == 1 else f"{name}.{module_id}"
        ledger = build_program_ledger(memoryview(path.read_bytes()))
        write_program_ledger(ledger, args.out / slug)
        print(f"== {slug} -> {args.out / slug}")
        print(render_peak_report(ledger, top=args.top))
        print()


if __name__ == "__main__":
    main()
