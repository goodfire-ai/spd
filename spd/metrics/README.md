# SPD Metrics Module

## Overview

This module implements a custom `Metric` base class (in `base.py`) that provides distributed metric computation without the complexity of torchmetrics (although it still uses much of the same API). All metrics inherit from this base class.

## Base Metric Class

The `Metric` base class provides:
- State registration via `add_state(name, default, dist_reduce_fx)`
- Distributed synchronization via `sync_dist()`
- Metric computation via `compute()`
- Device management via `.to(device)`
- State reset via `reset()`

### Key Methods

**`add_state(name, default, dist_reduce_fx)`**
- Registers a state variable that will be synchronized across ranks
- `dist_reduce_fx` can be:
  - `"sum"`: Gathers tensors from all ranks and sums them (for scalar metrics)
  - `"cat"`: Concatenates tensors from all ranks (for collecting samples)
- `default` must be a `Tensor` for "sum" or an empty `list` for "cat"

**`update(**kwargs)`**
- Accumulates metric state for each batch
- Called once per batch during training/evaluation
- Must be implemented by subclasses

**`compute()`**
- Computes the final metric value from current state
- Works on whatever state is available (local or synchronized)
- Must be implemented by subclasses
- **For training with DDP:** Call directly after updates to get per-rank metrics
- **For evaluation:** Call after `sync_dist()` to get global metrics

**`sync_dist()`**
- Synchronizes all registered states across distributed ranks
- For "sum" reduction: gathers and sums tensors from all ranks
- For "cat" reduction: concatenates local list, gathers from all ranks, then concatenates across ranks
- No-op if not in distributed mode
- **Must be called before `compute()` for evaluation to get global metrics**

**`reset()`**
- Resets all metric states to their default values
- Useful for reusing metric instances across multiple evaluation runs

**`.to(device)`**
- Moves all registered tensor states to the specified device
- For `Tensor` states: calls `.to(device)`
- For `list` states: moves all tensor elements in the list
- Returns `self` for method chaining

## Usage Patterns

### Evaluation Metrics

For evaluation metrics (in `eval.py`):
1. Call `update()` for each batch to accumulate state
2. Call `sync_dist()` to synchronize state across all ranks
3. Call `compute()` to get the final metric value

The framework automatically calls `sync_dist()` before `compute()` in evaluation mode.

Example:
```python
class MyEvalMetric(Metric):
    is_differentiable = False
    slow = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Register states that will be synchronized
        self.add_state("sum_value", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, batch, **kwargs):
        # Accumulate per-batch statistics
        self.sum_value += batch.sum()
        self.n_samples += batch.numel()

    def compute(self):
        # Compute final metric (works on local or synced state)
        return self.sum_value / self.n_samples

# Usage in evaluation:
metric = MyEvalMetric()
for batch in dataloader:
    metric.update(batch=batch)
metric.sync_dist()  # Synchronize across ranks
result = metric.compute()  # Get global metric
```

### Training Loss Metrics

For metrics used in training losses:
- Set `is_differentiable=True` to allow gradients to flow through
- Call `compute()` directly (without `sync_dist()`) to get per-rank loss
- DistributedDataParallel automatically syncs gradients during `loss.backward()`

Example:
```python
class MyTrainingLoss(Metric):
    is_differentiable = True
    slow = False

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.add_state("sum_loss", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_examples", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, batch, target, **kwargs):
        loss = some_differentiable_computation(batch, target)
        self.sum_loss += loss
        self.n_examples += batch.shape[0]

    def compute(self):
        # Returns differentiable tensor for backward pass
        return self.sum_loss / self.n_examples

# Usage in training:
metric = MyTrainingLoss(model)
metric.update(batch=batch, target=target)
loss = metric.compute()  # Get per-rank loss (no sync needed as DDP will sync gradients itself)
loss.backward()  # DDP syncs gradients automatically
```

## Metric Class Attributes

All metrics can set these class attributes:
- `is_differentiable: bool | None` - Whether the metric is differentiable (required for training losses)
- `slow: bool` - Whether the metric is slow to compute (for logging purposes)

## Distributed Behavior

### Shape Validation
Unlike torchmetrics, our implementation requires all tensor shapes to match exactly across ranks:
- If shapes don't match, an `AssertionError` is raised with detailed rank/shape information
- No automatic padding or trimming - this ensures consistent metric computation

### Non-Distributed Mode
When not running in distributed mode (`torch.distributed.is_available()` or `is_initialized()` returns False):
- `sync_dist()` becomes a no-op
- Metrics work identically in single-GPU or CPU mode

## Implementation Notes

### State Registration
States registered with `add_state()` are stored as instance attributes and tracked internally:
```python
self.add_state("my_values", default=[], dist_reduce_fx="cat")
# Now you can access self.my_values like any other attribute
self.my_values.append(some_tensor)
```

### Concatenation States
For states with `dist_reduce_fx="cat"`:
- Store as a list during `update()`: `self.my_state.append(tensor)`
- After `sync_dist()`, the state becomes a concatenated tensor
- `compute()` should handle both list and tensor cases (see `ci_histograms.py` for example)

Example:
```python
def compute(self):
    # Handle both list (before sync) and tensor (after sync)
    if isinstance(self.my_state, list):
        if len(self.my_state) == 0:
            raise ValueError("No samples collected")
        values = torch.cat(self.my_state, dim=0)
    else:
        values = self.my_state
    return values.mean()
```