# PPGD Source Shape Issues

Three related issues in how `PersistentPGDState` computes source tensor shapes,
primarily latent (only triggered if PPGD is used with MSE loss) but worth fixing
for correctness and robustness.

## Issue 1: `batch_dims` conflates batch and sequence for MSE

**Location**: `run_spd.py:239-256`

For KL (LM) inputs shaped `(B, S)`, `batch_dims = (B, S)` and the two args passed
to `PersistentPGDState.__init__` are `seq_len=S`, `batch_size=B`. Correct.

For MSE inputs shaped `(B, D)`, `batch_dims = sample_batch.shape[:-1] = (B,)`.
Then `seq_len=batch_dims[-1]=B` and `batch_size=batch_dims[0]=B` — both resolve
to the batch size. Source shapes become `(n, B, C)` instead of `(n, C)`, and the
subsequent `expand` in `get_mask_infos` would crash on a dimension mismatch
(`source.expand(B, -1)` on a 3-dim tensor).

**Fix**: Replace the separate `batch_size` and `seq_len` args with a single
`batch_dims: tuple[int, ...]`. Derive `source_leading_dims` from it:

```python
match cfg.scope:
    case SingleSourceScope():
        source_leading_dims = [1] * len(batch_dims)
    case BroadcastAcrossBatchScope():
        source_leading_dims = [1] + list(batch_dims[1:])
    case RepeatAcrossBatchScope(n_sources=n):
        source_leading_dims = [n] + list(batch_dims[1:])
    case PerBatchPerPositionScope():
        source_leading_dims = list(batch_dims)
```

For LM `(B, S)`: RepeatAcrossBatch produces `[n, S, C]` — unchanged.
For MSE `(B,)`: produces `[n, C]` — correct (no spurious seq dim).

The call site in `run_spd.py` simplifies to just passing `batch_dims` directly.

## Issue 2: `n_sources` must divide per-rank microbatch size

**Location**: `persistent_pgd.py:228`

The assertion `B % N == 0` in `get_mask_infos` fires at forward time with an
opaque message. `B` is the per-rank microbatch size (e.g. 8 with global
batch=64 on 8 GPUs), not the global batch size. A user setting `n_sources=16`
would pass the mental check (16 divides 64) but crash at runtime (16 doesn't
divide 8).

**Fix**: Validate at init time in `PersistentPGDState.__init__`, right after
the match block:

```python
case RepeatAcrossBatchScope(n_sources=n):
    assert batch_dims[0] % n == 0, (
        f"n_sources={n} must divide the per-rank microbatch size "
        f"{batch_dims[0]}, not the global batch size. "
        f"With DDP, reduce n_sources or use fewer ranks."
    )
```

This catches misconfiguration at startup with an actionable error message.

## Issue 3: Hardcoded 3-dim `repeat` in `get_mask_infos`

**Location**: `persistent_pgd.py:229`

```python
expanded_adv_sources[module_name] = source.repeat(B // N, 1, 1)
```

This hardcodes exactly 3 dimensions. The `expand` path (line 226) correctly
unpacks `*batch_dims`, but the `repeat` path would break for 2-dim sources
(which issue 1's fix would produce for MSE).

**Fix**: Derive the repeat tuple from the source's actual ndim:

```python
if N == 1 or N == B:
    expanded_adv_sources[module_name] = source.expand(*batch_dims, -1)
else:
    assert B % N == 0, ...
    repeat_dims = (B // N,) + (1,) * (source.ndim - 1)
    expanded_adv_sources[module_name] = source.repeat(*repeat_dims)
```

Works for both 2-dim `(N, C)` and 3-dim `(N, S, C)` sources.

## Dependencies

Fix 1 is a prerequisite for fix 3 — without correctly-shaped sources from init,
generic expansion doesn't help. Fix 2 is independent but naturally fits into the
same `__init__` refactor.

Files touched: `spd/persistent_pgd.py` (init + get_mask_infos), `spd/run_spd.py` (call site).
