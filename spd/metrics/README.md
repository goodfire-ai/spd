# SPD Metrics Module

## Evaluation Metrics

For evaluation metrics:
- Set `sync_on_compute=True` - We want the `Metric.compute()` to be calculated on the data that has been gathered across ranks
- `is_differentiable` attribute can be ignored for eval metrics since we don't do backward passes
- `Metric.update()` processes each batch individually
- `Metric.compute()` the inputs to this are method are pre-synced across ranks (via a sneaky attribute on the baseclass: `Metric.compute: Callable = self._wrap_compute(self.compute)`)

## Metric Calling Behavior

### `Metric.__call__()` Internals
- Calls both `update()` and `compute()` sequentially
- We set `full_state_update=False` on all metrics to ensure `update()` is only called once

### Why `full_state_update` Matters
When `full_state_update=True` (default torchmetrics behavior), `Metric._forward_full_state_update` is called. Here,
- `update()` is called **twice** 
- First call: uses the Metric state that has been accumulated from previous batches (if there were any). This allows for updating the global state of the Metric correctly.
- Second call: wipes the Metric state before running.
- The output of forward pass will be the output of the second call, but the state of the Metric will be calculated based on the first call.

When `full_state_update=False` (we set this in all of our Metrics):
- `update()` is called once
- Computes per-batch output while ignoring all previously accumulated state (just like the second forward pass when `full_state_update=True`)

> **TODO**: This behavior is confusing. I'm considering creating `SPDMetric` which inherits from `Metric` and raises an error when `forward()` is called directly to prevent people worrying about the above.

## Training Loss Metrics

Requirements for metrics used in training losses:
- **Must** set `is_differentiable=True`
- Set `sync_on_compute=False` to avoid double-syncing (DistributedDataParallel automatically syncs gradients during `loss.backward()`)