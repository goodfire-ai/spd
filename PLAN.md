## Consolidated Metrics and Losses: Implementation Plan

### Goal
- Unify evaluation metrics and training losses under a single, TorchMetrics-like interface so that:
  - Existing eval metrics adopt a `Metric` base class (rename from `StreamingEval`).
  - A `forward()` convenience method exists on the base class.
  - Later, losses can be exposed as metrics and logged by default during eval (without config boilerplate).

### Non-goals (for the first iteration)
- Cross-rank reduction semantics inside `Metric.compute()` itself (will rely on existing outer reduction).
- Refactor training to use `Metric` for losses (this comes after we’re happy with the interface).

---

## Phase 1 — Rename interface and add `forward()`
Make the minimal changes needed to land a solid `Metric` interface and preserve behavior.

- TODO: Rename `StreamingEval` to `Metric` in `spd/eval.py`.
- TODO: Keep `SLOW: ClassVar[bool]` unchanged; default False where appropriate.
- TODO: Add `forward()` to base class that calls `watch_batch()` then `compute()` and returns the result.
- TODO: Ensure all concrete metric classes subclass `Metric` (update class declarations).
- TODO: Keep `evaluate()` streaming semantics: keep calling `watch_batch()` per step and `compute()` once at the end.
- TODO: Update `EVAL_CLASSES` mapping to reflect rename; ensure zero behavior change for current configs.
- TODO: Retain backward compat: provide a temporary alias so configs using previous class names still resolve (optional deprecation warning).

Interface sketch (no behavior change expected):

```python
class Metric(ABC):
    SLOW: ClassVar[bool]

    @abstractmethod
    def __init__(self, model: ComponentModel, config: Config, **kwargs: Any) -> None: ...

    @abstractmethod
    def watch_batch(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
    ) -> None: ...

    @abstractmethod
    def compute(self) -> Mapping[str, float | Image.Image]: ...

    def forward(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
    ) -> Mapping[str, float | Image.Image]:
        self.watch_batch(batch=batch, target_out=target_out, ci=ci)
        return self.compute()
```

Notes:
- `evaluate()` will continue using streaming behavior (call `watch_batch()` N times then `compute()` once). `forward()` is a convenience method mostly for parity with TorchMetrics and potential future callers.

---

## Phase 2 — Default loss metrics during eval (wrapper integration)
Introduce default loss metrics for eval that do not require explicit config entries.

- TODO: Add a metric wrapper (e.g., `LossTermsMetric`) that internally calls `calculate_losses(...)` per batch and keeps running means for each loss term.
- TODO: Ensure this metric is always included at eval-time when enabled, regardless of `config.eval_metrics`.
- TODO: Aggregate per-term means across `n_eval_steps` and return keys like `loss/faithfulness`, `loss/recon`, `loss/total`, etc.
- TODO: Decide whether to also report weighted totals (respecting coefficients) vs. raw component values; likely both: `loss/total_weighted` and `loss/*` raw.
- TODO: Add a config flag to allow disabling this default (e.g., `include_loss_metrics_in_eval: bool = True`).

Implementation sketch:
- Instantiate `LossTermsMetric` along with configured metrics in `evaluate()` when the flag is enabled.
- Compute `weight_deltas` once per eval window if needed and reuse in the metric to avoid redundant computation.
- Use existing cross-rank averaging outside of the metric (unchanged behavior for now).

---

## Phase 3 — Migrate losses to Metric classes
Once the interface feels right, refactor loss computation to be first-class `Metric`s.

- TODO: Introduce concrete `Metric` subclasses for core losses (e.g., `FaithfulnessLossMetric`, `ReconstructionLossMetric`, etc.).
- TODO: Replace the `LossTermsMetric` wrapper with dedicated metrics or keep it as a convenience aggregator.
- TODO: Introduce a `loss_metrics` section in the config if we want fine-grained control; otherwise, keep the default-on behavior with an allowlist/blocklist.
- TODO: Update training to optionally use these metrics for logging (to unify naming and aggregation).

---

## Phase 4 — DDP-aware compute (optional, later)
Add an optional reduction mode to `Metric.compute()` so each metric can specify how to reduce across ranks.

- TODO: Define a simple protocol for reduction (e.g., `none`, `mean`, `sum`) per key.
- TODO: Keep images and non-reducible outputs excluded from reduction by default.
- TODO: Integrate with `spd.utils.distributed_utils` helpers.

---

## Phase 5 — Backward compatibility and migration

- EDIT: IGNORE ALL BACKWARD COMPATIBILITY ISSUES

## Phase 6 — Tests

- TODO: Unit test `forward()`, `watch_batch()`, `compute()` lifecycle for a simple metric.
- TODO: Test `LossTermsMetric` aggregation over multiple batches.
- TODO: Smoke test `evaluate()` still logs existing metrics and additionally loss metrics when enabled.
- TODO: Test distributed averaging path stays intact (outer reduction only for now).

---

## Phase 7 — Documentation

- TODO: Add README section describing the `Metric` interface (constructor, `watch_batch`, `compute`, `forward`).
- TODO: Document default loss metrics behavior and config flag to disable.
- TODO: Changelog: interface rename, new default eval losses.

---

## Open decisions to confirm
1) File layout: keep metrics in `spd/eval.py` or move to `spd/metrics.py` now? (Leaning to keep file for minimal diff, rename classes only.)
2) Back-compat: Maintain aliases for old class names in `EVAL_CLASSES`? If yes, warn once.
3) Loss reporting: Prefer logging both raw per-term values and a `loss/total_weighted` using config coefficients?
4) Default-on behavior: Add `include_loss_metrics_in_eval: bool = True` to `Config`? Name OK?
5) `forward()` semantics: OK that `forward()` returns current aggregate (i.e., computes after updating internal state), mirroring TorchMetrics?
6) Any eval-time computational budget constraints (e.g., reusing `weight_deltas` across steps, skipping SLOW loss metrics by default)?




