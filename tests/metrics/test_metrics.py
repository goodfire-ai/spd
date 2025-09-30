"""Unit tests for custom Metric class (single process)."""

from typing import Any, override

import pytest
import torch

from spd.metrics.base import Metric


class SumMetric(Metric):
    """Simple metric that computes mean using sum reduction."""

    total: torch.Tensor
    count: torch.Tensor

    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(self, *, value: torch.Tensor, **_: Any) -> None:
        self.total += value.sum()
        self.count += value.numel()

    @override
    def compute(self) -> float:
        return (self.total / self.count).item()


class CatMetric(Metric):
    """Simple metric that concatenates values using cat reduction."""

    values: list[torch.Tensor] | torch.Tensor

    def __init__(self):
        super().__init__()
        self.add_state("values", default=[], dist_reduce_fx="cat")

    @override
    def update(self, *, value: torch.Tensor, **_: Any) -> None:
        assert isinstance(self.values, list)
        self.values.append(value)

    @override
    def compute(self) -> torch.Tensor:
        if isinstance(self.values, list):
            return torch.cat(self.values, dim=0)
        return self.values


class TestMetricBasics:
    """Test basic metric functionality in single process."""

    def test_add_state_sum(self):
        """Test add_state with sum reduction."""
        metric = SumMetric()
        assert hasattr(metric, "total")
        assert hasattr(metric, "count")
        assert "total" in metric._state_names
        assert "count" in metric._state_names
        assert metric._state_reduce_fns["total"] == "sum"
        assert metric._state_reduce_fns["count"] == "sum"

    def test_add_state_cat(self):
        """Test add_state with cat reduction."""
        metric = CatMetric()
        assert hasattr(metric, "values")
        assert "values" in metric._state_names
        assert metric._state_reduce_fns["values"] == "cat"

    def test_add_state_invalid_reduce(self):
        """Test that invalid reduce function raises error."""

        class BadMetric(Metric):
            def __init__(self):
                super().__init__()
                self.add_state("bad", default=torch.tensor(0.0), dist_reduce_fx="invalid")  # pyright: ignore[reportArgumentType]

            @override
            def update(self, **_: Any):
                pass

            @override
            def compute(self):
                pass

        with pytest.raises(AssertionError, match="Invalid reduce function"):
            BadMetric()

    def test_add_state_type_mismatch(self):
        """Test that mismatched default type raises error."""

        class BadSumMetric(Metric):
            def __init__(self):
                super().__init__()
                # Sum requires tensor, not list
                self.add_state("bad", default=[], dist_reduce_fx="sum")

            @override
            def update(self, **_: Any):
                pass

            @override
            def compute(self):
                pass

        class BadCatMetric(Metric):
            def __init__(self):
                super().__init__()
                # Cat requires empty list, not tensor
                self.add_state("bad", default=torch.tensor(0.0), dist_reduce_fx="cat")

            @override
            def update(self, **_: Any):
                pass

            @override
            def compute(self):
                pass

        with pytest.raises(AssertionError, match="sum reduce requires Tensor"):
            BadSumMetric()

        with pytest.raises(AssertionError, match="cat reduce requires empty list"):
            BadCatMetric()

    def test_update_and_compute_sum(self):
        """Test update and compute for sum reduction."""
        metric = SumMetric()

        # Update with some values
        metric.update(value=torch.tensor([1.0, 2.0, 3.0]))
        metric.update(value=torch.tensor([4.0, 5.0]))

        # Compute mean
        result = metric.compute()
        expected = (1 + 2 + 3 + 4 + 5) / 5
        assert result == pytest.approx(expected)

    def test_update_and_compute_cat(self):
        """Test update and compute for cat reduction."""
        metric = CatMetric()

        # Update with some values
        metric.update(value=torch.tensor([1.0, 2.0]))
        metric.update(value=torch.tensor([3.0, 4.0, 5.0]))

        # Compute concatenated result
        result = metric.compute()
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        torch.testing.assert_close(result, expected)

    def test_reset_sum(self):
        """Test reset for sum reduction."""
        metric = SumMetric()

        # Update
        metric.update(value=torch.tensor([1.0, 2.0]))
        assert metric.total != 0.0
        assert metric.count != 0

        # Reset
        metric.reset()
        assert metric.total == 0.0
        assert metric.count == 0

    def test_reset_cat(self):
        """Test reset for cat reduction."""
        metric = CatMetric()

        # Update
        metric.update(value=torch.tensor([1.0, 2.0]))
        assert len(metric.values) > 0

        # Reset
        metric.reset()
        assert len(metric.values) == 0

    def test_to_device(self):
        """Test moving metric to CPU.

        NOTE: If GPU is not available, this test will be trivial, but we run it anyway
        """
        device = "cpu" if torch.cuda.is_available() else "cuda"
        metric = SumMetric()
        metric.update(value=torch.tensor([1.0, 2.0]))

        metric.to(device)
        assert metric.total.device.type == device
        assert metric.count.device.type == device

    def test_sync_dist_noop_single_process(self):
        """Test that sync_dist is a no-op in single process mode."""
        metric = SumMetric()
        metric.update(value=torch.tensor([1.0, 2.0, 3.0]))

        total_before = metric.total.clone()
        count_before = metric.count.clone()

        metric.sync_dist()

        torch.testing.assert_close(metric.total, total_before)
        torch.testing.assert_close(metric.count, count_before)

    def test_multiple_states(self):
        """Test metric with multiple states."""

        class MultiStateMetric(Metric):
            sum_state: torch.Tensor
            cat_state: list[torch.Tensor] | torch.Tensor

            def __init__(self):
                super().__init__()
                self.add_state("sum_state", default=torch.tensor(0.0), dist_reduce_fx="sum")
                self.add_state("cat_state", default=[], dist_reduce_fx="cat")

            @override
            def update(self, *, value: torch.Tensor, **_: Any) -> None:
                self.sum_state += value.sum()
                assert isinstance(self.cat_state, list)
                self.cat_state.append(value)

            @override
            def compute(self) -> tuple[float, torch.Tensor]:
                assert isinstance(self.cat_state, list)
                return self.sum_state.item(), torch.cat(self.cat_state, dim=0)

        metric = MultiStateMetric()
        metric.update(value=torch.tensor([1.0, 2.0]))
        metric.update(value=torch.tensor([3.0]))

        sum_result, cat_result = metric.compute()
        assert sum_result == 6.0
        torch.testing.assert_close(cat_result, torch.tensor([1.0, 2.0, 3.0]))
