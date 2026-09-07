"""The trace gate is a `python -m` composition root, so it owns its process's import order.

tokamax must load before pyarrow: xprof's C++ static initializer (tokamax's dependency)
self-deadlocks when pyarrow's bundled protobuf registered first — the guard `run.py` and
the root conftest carry — and the gate's eval imports reach pyarrow through the LM data
path while the routed experts import tokamax lazily at trace time. `sys.modules` is
insertion-ordered, so a fresh interpreter that imports the module records which of the two
began loading first. A subprocess, because this test process has both loaded already."""

import os
import subprocess
import sys

_PROBE = """
import sys

import param_decomp.experiments.lm.trace_check

loaded = list(sys.modules)
assert "tokamax" in loaded, "the trace gate must import tokamax"
assert "pyarrow" in loaded, "the trace gate's eval imports reach pyarrow"
assert loaded.index("tokamax") < loaded.index("pyarrow"), (
    loaded.index("tokamax"),
    loaded.index("pyarrow"),
)
"""


def test_trace_check_imports_tokamax_before_pyarrow() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=os.environ | {"JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-3000:]
