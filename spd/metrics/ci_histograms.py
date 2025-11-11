from collections import defaultdict
from typing import Any, ClassVar, override

import torch
from PIL import Image
from torch import Tensor

from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.plotting import plot_ci_values_histograms_from_counts
from spd.utils.distributed_utils import gather_all_tensors


class CIHistograms(Metric):
    N_BINS: ClassVar[int] = 100

    slow: ClassVar[bool] = True
    metric_section: ClassVar[str] = "figures"

    def __init__(
        self,
        model: ComponentModel,
        n_batches_accum: int | None = None,
    ):
        self.n_batches_accum = n_batches_accum
        self.batches_seen = 0

        # Store histogram counts instead of raw tensors - much smaller memory footprint
        # Each update bins values incrementally, avoiding expensive concatenation
        self.lower_leaky_hist_counts = defaultdict[str, Tensor](lambda: torch.zeros(self.N_BINS))
        self.pre_sigmoid_hist_counts = defaultdict[str, Tensor](lambda: torch.zeros(self.N_BINS))

    @override
    def update(self, *, ci: CIOutputs, **_: Any) -> None:
        if self.n_batches_accum is not None and self.batches_seen >= self.n_batches_accum:
            return

        self.batches_seen += 1

        # Bin values incrementally: O(1) binning per update, no concatenation needed
        # Detach to avoid keeping computation graph alive
        for k, v in ci.lower_leaky.items():
            v_detached = v.detach().flatten()
            hist_counts = torch.histc(v_detached, bins=self.N_BINS, min=0.0, max=1.0).long()
            self.lower_leaky_hist_counts[k] += hist_counts

        for k, v in ci.pre_sigmoid.items():
            v_detached = v.detach().flatten()
            hist_counts = torch.histc(v_detached, bins=self.N_BINS, min=0.0, max=1.0).long()
            self.pre_sigmoid_hist_counts[k] += hist_counts

    @override
    def compute(self) -> dict[str, Image.Image]:
        if self.batches_seen == 0:
            raise RuntimeError("No batches seen yet")

        # Gather histogram counts from all ranks and sum them
        # Much smaller data transfer: just 100 integers per module instead of millions of floats
        lower_leaky_hist_counts: dict[str, Tensor] = {}
        for module_name, local_counts in self.lower_leaky_hist_counts.items():
            gathered = gather_all_tensors(local_counts.float())  # Convert to float for gathering
            # Sum counts across all ranks
            lower_leaky_hist_counts[module_name] = torch.stack(gathered, dim=0).sum(dim=0).long()

        pre_sigmoid_hist_counts: dict[str, Tensor] = {}
        for module_name, local_counts in self.pre_sigmoid_hist_counts.items():
            gathered = gather_all_tensors(local_counts.float())  # Convert to float for gathering
            # Sum counts across all ranks
            pre_sigmoid_hist_counts[module_name] = torch.stack(gathered, dim=0).sum(dim=0).long()

        lower_leaky_fig = plot_ci_values_histograms_from_counts(lower_leaky_hist_counts, self.N_BINS)
        pre_sigmoid_fig = plot_ci_values_histograms_from_counts(pre_sigmoid_hist_counts, self.N_BINS)

        return {
            "causal_importance_values": lower_leaky_fig,
            "causal_importance_values_pre_sigmoid": pre_sigmoid_fig,
        }
