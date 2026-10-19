"""End-to-end pins for the memory ledger: a real (tiny) XLA compile's proto dump parses
into a ledger whose accounting adds up, survives the CSV round-trip, and diffs by
provenance. CPU-only — the parsers are backend-blind (the same HloProto schema serves
the deviceless GPU-pipeline dumps the fit check emits)."""

from pathlib import Path

import jax
import jax.numpy as jnp

from param_decomp.core.tools.ledger_diff import diff_programs
from param_decomp.core.tools.memory_ledger import (
    AllocationKind,
    ProgramLedger,
    emit_program_ledger,
    read_program_ledger,
)


def _emitted_ledger(tmp_path: Path, name: str, width: int) -> ProgramLedger:
    def f(x: jax.Array, w: jax.Array) -> jax.Array:
        return (jnp.tanh(x @ w) @ w.T).sum()

    x = jax.ShapeDtypeStruct((8, width), jnp.float32)
    w = jax.ShapeDtypeStruct((width, width), jnp.float32)
    dump_dir = tmp_path / name / "dump"
    options = {"xla_dump_to": str(dump_dir), "xla_dump_hlo_as_proto": True}
    jax.jit(jax.grad(f, argnums=1), compiler_options=options).lower(x, w).compile()
    return emit_program_ledger(dump_dir, tmp_path / name)


def test_ledger_accounts_and_round_trips(tmp_path: Path) -> None:
    ledger = _emitted_ledger(tmp_path, "a", 16)
    assert ledger.module_name == "jit_f"

    arena_owners = [
        r for r in ledger.rows if r.bytes_owner and r.allocation_kind is AllocationKind.ARENA
    ]
    assert arena_owners
    for row in arena_owners:
        assert row.born_event is not None and row.dies_event is not None
        assert row.born_event < row.dies_event
        assert row.offset + row.size_bytes <= ledger.arena_bytes
    # The peak identity, re-derived from the rows alone (build asserts it internally too).
    live_at_peak = [
        r
        for r in arena_owners
        if r.born_event is not None
        and r.dies_event is not None
        and r.born_event <= ledger.peak_event < r.dies_event
    ]
    assert sum(r.size_bytes for r in live_at_peak) == ledger.peak_bytes

    assert any("jit(f)/" in r.op_path for r in ledger.rows)
    assert any("test_memory_ledger.py:" in r.source for r in ledger.rows)

    assert read_program_ledger(tmp_path / "a") == ledger
    report = (tmp_path / "a" / "peak_report.txt").read_text()
    assert "arena peak:" in report and ledger.peak_instruction in report


def test_ledger_diff_groups_by_provenance(tmp_path: Path) -> None:
    small = _emitted_ledger(tmp_path, "small", 16)
    large = _emitted_ledger(tmp_path, "large", 64)

    self_diff = diff_programs("jit_f", small, small, peak_only=False)
    assert not (self_diff.created or self_diff.destroyed or self_diff.resized)

    diff = diff_programs("jit_f", small, large, peak_only=False)
    assert diff.peak_bytes[1] > diff.peak_bytes[0]
    changed = diff.created + diff.destroyed + diff.resized
    assert changed
    assert all(d.key for d in changed)
    assert sum(d.delta_bytes for d in changed) > 0
