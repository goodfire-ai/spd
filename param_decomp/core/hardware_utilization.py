"""Executed-FLOPs hardware utilization (HFU) for the jitted train step.

XLA's post-optimization cost analysis counts the flops the compiled per-device program
actually EXECUTES — rematerialization recompute included — so the ratio to hardware peak
is hardware utilization (HFU), not model utilization (MFU, which counts each model flop
once regardless of recompute).

The device kinds with a declared peak are a CLOSED enumeration (`DeviceKind`). A fleet is
checked against it at the launch boundary (`sharding.initialize_topology`, the fit
check's described topology) — before the frozen target loads and the multi-minute step
compile — so an unlisted kind refuses a run in its first seconds, never after allocation
at the first measured step. The step-cost path re-checks as a tripwire.
"""

import dataclasses
from collections.abc import Sequence
from typing import Any, Literal, cast, get_args

import jax

DeviceKind = Literal["NVIDIA B200", "NVIDIA H200", "NVIDIA H100 80GB HBM3", "cpu"]
"""The exact `jax.Device.device_kind` strings with a declared bf16 dense peak (or the
deliberate no-peak `cpu` arm). Adding a kind = adding its literal here AND its row in
`peak_bf16_dense_flops_per_second`; the match below is exhaustive over the literal."""

DEVICE_KINDS: frozenset[str] = frozenset(get_args(DeviceKind))


def checked_device_kind(devices: Sequence[Any]) -> DeviceKind:
    """The one `DeviceKind` a homogeneous fleet runs on, refusing heterogeneity and any
    kind outside the enumeration with the remedy named (`jax.Device` is not a static
    type in jax 0.10 — hence the loose element type)."""
    kinds = {device.device_kind for device in devices}
    assert len(kinds) == 1, f"heterogeneous device kinds: {sorted(kinds)}"
    (kind,) = kinds
    assert kind in DEVICE_KINDS, (
        f"device kind {kind!r} has no declared bf16 dense peak — add it to "
        f"hardware_utilization.DeviceKind and its datasheet row to "
        f"peak_bf16_dense_flops_per_second (known: {sorted(DEVICE_KINDS)})"
    )
    return cast(DeviceKind, kind)


def peak_bf16_dense_flops_per_second(device_kind: DeviceKind) -> float | None:
    """Dense (non-sparsity) bf16 tensor-core peak of ONE device, from the vendor datasheet.

    `None` is the deliberate no-peak arm (CPU test/toy runs) — HFU is not emitted there."""
    match device_kind:
        case "NVIDIA B200":
            return 2.25e15  # HGX/DGX B200 datasheets: 4.5 PFLOPS/GPU with sparsity, dense = half
        case "NVIDIA H200":
            return 989.4e12  # same Hopper SXM compute die as the H100, memory is the delta
        case "NVIDIA H100 80GB HBM3":
            return 989.4e12  # H100 SXM datasheet: 1,979 TFLOPS bf16 with sparsity, dense = half
        case "cpu":
            return None


@dataclasses.dataclass(frozen=True)
class StepCost:
    """One compiled train step's executed-FLOPs cost, fixed at compile time."""

    flops_per_step: float
    """Whole-run executed flops of one step: the per-device SPMD program's post-optimization
    cost-analysis flops × n_devices (every device runs the same partitioned program)."""

    n_devices: int
    peak_flops_per_device: float | None

    @classmethod
    def of(cls, compiled: jax.stages.Compiled, devices: Sequence[Any]) -> "StepCost":
        """`devices` is the run's whole fleet, `jax.devices()`. Its kind was checked at the
        launch boundary; `checked_device_kind` here is the tripwire."""
        kind = checked_device_kind(devices)
        analysis = compiled.cost_analysis()
        assert isinstance(analysis, dict) and "flops" in analysis, (
            f"the compiled train step exposes no flops cost analysis (got {analysis!r}); "
            "HFU cannot be measured on this backend/path — investigate, don't skip the metric"
        )
        return cls(
            flops_per_step=float(analysis["flops"]) * len(devices),
            n_devices=len(devices),
            peak_flops_per_device=peak_bf16_dense_flops_per_second(kind),
        )

    def hfu(self, step_time_s: float) -> float | None:
        """Executed flops over peak flops for one measured step; None on a no-peak device."""
        if self.peak_flops_per_device is None:
            return None
        return self.flops_per_step / (step_time_s * self.n_devices * self.peak_flops_per_device)
