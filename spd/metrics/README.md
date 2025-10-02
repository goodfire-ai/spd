# SPD Metrics Module

## Overview

This module implements metrics for training and evaluation. All metrics implement the `MetricInterface` and handle distributed synchronization directly in their implementation using `all_reduce()` or `gather_all_tensors()`.

## MetricInterface

The `MetricInterface` (in `base.py`) provides a simple interface:

```python
class MetricInterface(ABC):
    @abstractmethod
    def update(
        self,
        batch: Tensor,
        target_out: Tensor,
        ci: dict[str, Tensor],
        current_frac_of_training: float,
        ci_upper_leaky: dict[str, Tensor],
        weight_deltas: dict[str, Tensor],
    ) -> None:
        """Update metric state with a batch of data."""
        pass

    @abstractmethod
    def compute(self) -> Any:
        """Compute the final metric value(s)."""
        pass
```

## Implementation Pattern

All metrics follow a "flat" pattern where distributed logic is inlined directly in the metric implementation. This makes the distributed behavior explicit and easy to understand.

### Training Loss Metrics (2-Scalar Pattern)

For metrics that accumulate a sum and count:

```python
class MyTrainingLoss(MetricInterface):
    def __init__(self, model: ComponentModel, ...):
        self.model = model
        device = get_device()
        self.sum_loss = torch.tensor(0.0, device=device)
        self.n_examples = torch.tensor(0, device=device)

    def update(self, batch, target_out, ci, ...):
        loss, n_examples = _my_loss_update(...)
        self.sum_loss += loss
        self.n_examples += n_examples

    def compute(self):
        # Synchronize across ranks
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return sum_loss / n_examples
```

Examples: `FaithfulnessLoss`, `StochasticReconLoss`, `CIMaskedReconLoss`

### Dict State Metrics

For metrics that track multiple values in a dictionary:

```python
class MyDictMetric(MetricInterface):
    def __init__(self, model: ComponentModel, ...):
        self.model = model
        device = get_device()
        self.loss_sums = {
            "loss_a": torch.tensor(0.0, device=device),
            "loss_b": torch.tensor(0.0, device=device),
        }
        self.n_examples = torch.tensor(0, device=device)

    def update(self, batch, target_out, ci, ...):
        self.loss_sums["loss_a"] += compute_loss_a(...)
        self.loss_sums["loss_b"] += compute_loss_b(...)
        self.n_examples += batch.shape[0]

    def compute(self):
        # Reduce all dict values
        reduced = {
            key: all_reduce(val, op=ReduceOp.SUM).item()
            for key, val in self.loss_sums.items()
        }
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM).item()
        return {key: val / n_examples for key, val in reduced.items()}
```

Examples: `CEandKLLosses`, `CIMeanPerComponent`

### Cat Reduction Metrics

For metrics that collect samples across batches, choose the appropriate reduction:

**If you need all raw data** (e.g., for plotting), use `all_cat()`:

```python
class MyCatMetric(MetricInterface):
    def __init__(self, model: ComponentModel, ...):
        self.samples: list[Tensor] = []

    def update(self, batch, target_out, ci, ...):
        self.samples.append(ci["layer"])

    def compute(self):
        all_samples = all_cat(self.samples)  # Gathers all data from all ranks
        return plot_histogram(all_samples)
```

Example: `CIHistograms`

**If you only need aggregate statistics** (e.g., mean), use sum+count reduction (more efficient):

```python
class MyStatMetric(MetricInterface):
    def __init__(self, model: ComponentModel, ...):
        self.samples: list[Tensor] = []

    def update(self, batch, target_out, ci, ...):
        self.samples.append(ci["layer"])

    def compute(self):
        local_tensor = torch.cat(self.samples, dim=0)
        local_sum = local_tensor.sum()
        local_count = torch.tensor(local_tensor.numel(), device=get_device())
        global_sum = all_reduce(local_sum, op=ReduceOp.SUM)
        global_count = all_reduce(local_count, op=ReduceOp.SUM)
        return (global_sum / global_count).item()
```

Example: `CI_L0`

### Dynamic State Metrics

For metrics with dynamically-keyed state:

```python
class MyDynamicMetric(MetricInterface):
    def __init__(self, ...):
        self.states: defaultdict[str, list[float]] = defaultdict(list)

    def update(self, batch, target_out, ci, ...):
        for key, value in compute_values(...).items():
            self.states[key].append(value)

    def compute(self):
        results: dict[str, float] = {}
        for key, vals in self.states.items():
            total_sum = all_reduce(torch.tensor(sum(vals)), op=ReduceOp.SUM).item()
            total_count = all_reduce(torch.tensor(len(vals)), op=ReduceOp.SUM).item()
            results[key] = total_sum / total_count
        return results
```

Examples: `StochasticReconSubsetCEAndKL`

## Distributed Behavior

When not running in distributed mode:
- `all_reduce()` and `gather_all_tensors()` return the input unchanged
- Metrics work identically in single-GPU or CPU mode

## Implementation Notes

### State Initialization

Always use `get_device()` to ensure tensors are on the correct device:

```python
device = get_device()
self.sum_loss = torch.tensor(0.0, device=device)
self.n_examples = torch.tensor(0, device=device)
```

### Functional Forms

Most metrics also provide functional forms (non-class versions) for one-off computations:

```python
# Functional form
loss = my_loss(model=model, batch=batch, ...)

# Class form (for accumulation across batches)
metric = MyLoss(model=model, ...)
metric.update(batch=batch, ...)
result = metric.compute()
```

### Reduction Operations

- Use `ReduceOp.SUM` for accumulating values across ranks (never use `ReduceOp.AVG` - it gives incorrect results when ranks have different sample counts)
- Use `all_cat(tensors)` for concatenating tensors locally and across ranks (replaces the `cat -> gather -> cat` pattern)
- Always reduce both the sum and count separately, then divide

## Usage in Evaluation

The evaluation framework (`eval.py`) calls metrics as follows:

```python
# Initialize metrics
metrics = [init_metric(cfg, model, ...) for cfg in metric_configs]

# Accumulate over batches
for batch in eval_iterator:
    for metric in metrics:
        metric.update(batch=batch, ...)

# Compute results (distributed reduction happens here)
for metric in metrics:
    result = metric.compute()  # Handles all_reduce internally
```

Note: The old `sync_dist()` pattern is no longer used. All distributed synchronization happens directly in `compute()`.
