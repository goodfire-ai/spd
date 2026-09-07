# param_decomp.routed — architecture notes

JAX expert-parallel routed compute for MoE layers: static jobs schedules (`RoutedJobs`
for a single shard, `ExpertShardedJobs` for the explicit `(data, tp)` mesh), the
sort/gather/unsort primitives with custom VJPs (transposes stay gathers, never XLA
scatter-adds — except the honestly-partial block gather), and the grouped-matmul backend
arms (`GroupedMatmulBackend`: `ragged_dot` | `tokamax` | `tokamax_split_vjp`). One
module, `experts.py`; its docstring is the design document.

## Public surface

The names consumers import today:

- Schedules: `RoutedJobs` / `routed_jobs`, `ExpertShardedJobs` / `expert_sharded_jobs`.
- Config vocabulary: `GroupedMatmulBackend`.
- Local primitives: `gather_tokens`, `sort_jobs`, `unsort_jobs`, `scatter_jobs`,
  `gather_job_blocks`, `sort_job_values`, `sum_jobs`, `combine_jobs`, `grouped_matmul`,
  `transposed_grouped_matmul`.
- Expert-parallel siblings: `ep_gather_tokens`, `ep_sort_jobs`, `ep_unsort_jobs`,
  `ep_gather_job_blocks`, `ep_sort_job_values`, `ep_sum_jobs`, `ep_combine_jobs`,
  `ep_grouped_matmul`.

## Layering

A leaf of the `param_decomp` layer graph, pinned by
`tests/core/test_runtime_standalone.py`: it imports jax (pallas included), jaxtyping,
and tokamax only — no `param_decomp` modules. `_`-prefixed names are module-private;
consumers import only the public surface above. Its tests live at `tests/routed/`.
