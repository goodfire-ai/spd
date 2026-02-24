from typing import Any, ClassVar, Literal, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import PersistentPGDReconLossConfig, PersistentPGDReconSubsetLossConfig
from spd.metrics.base import Metric
from spd.metrics.hidden_acts_recon_loss import calc_hidden_acts_mse, compute_per_module_metrics
from spd.models.component_model import CIOutputs, ComponentModel
from spd.persistent_pgd import PersistentPGDState, PPGDSources, get_ppgd_mask_infos
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import calc_sum_recon_loss_lm


class PPGDEvalLosses(Metric):
    """Eval losses using persistent PGD masks: hidden activation MSE and output reconstruction.

    Takes the full ppgd_states dict and evaluates all states in a single metric instance,
    sharing the target_acts forward pass across states.
    """

    slow: ClassVar[bool] = True
    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        ppgd_states: dict[
            PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig, PersistentPGDState
        ],
        use_delta_component: bool,
        output_loss_type: Literal["mse", "kl"],
    ) -> None:
        assert ppgd_states, "PPGDEvalLosses requires persistent PGD states"
        self.model = model
        self.use_delta_component = use_delta_component
        self.output_loss_type: Literal["mse", "kl"] = output_loss_type
        self.device = device

        use_suffix = len(ppgd_states) > 1
        self._entries: list[tuple[str, PPGDSources]] = []
        for cfg, state in ppgd_states.items():
            suffix = f"_{type(cfg).__name__}" if use_suffix else ""
            self._entries.append((suffix, state.get_effective_sources()))

        self._per_entry_module_sum_mse: list[dict[str, Tensor]] = [{} for _ in self._entries]
        self._per_entry_module_n: list[dict[str, Tensor]] = [{} for _ in self._entries]
        self._per_entry_output_sum_loss: list[Tensor] = [
            torch.tensor(0.0, device=device) for _ in self._entries
        ]
        self._per_entry_output_n: list[Tensor] = [
            torch.tensor(0, device=device) for _ in self._entries
        ]

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        ci: CIOutputs,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
        target_out: Float[Tensor, "..."],
        **_: Any,
    ) -> None:
        target_acts = self.model(batch, cache_type="output").cache
        batch_dims = next(iter(ci.lower_leaky.values())).shape[:-1]

        for i, (_suffix, effective_sources) in enumerate(self._entries):
            mask_infos = get_ppgd_mask_infos(
                ci=ci.lower_leaky,
                weight_deltas=weight_deltas if self.use_delta_component else None,
                ppgd_sources=effective_sources,
                routing_masks="all",
                batch_dims=batch_dims,
            )
            per_module, comp_output = calc_hidden_acts_mse(
                model=self.model,
                batch=batch,
                mask_infos=mask_infos,
                target_acts=target_acts,
            )
            for key, (mse, n) in per_module.items():
                if key not in self._per_entry_module_sum_mse[i]:
                    self._per_entry_module_sum_mse[i][key] = torch.tensor(0.0, device=self.device)
                    self._per_entry_module_n[i][key] = torch.tensor(0, device=self.device)
                self._per_entry_module_sum_mse[i][key] += mse.detach()
                self._per_entry_module_n[i][key] += n

            output_loss = calc_sum_recon_loss_lm(
                pred=comp_output, target=target_out, loss_type=self.output_loss_type
            )
            self._per_entry_output_sum_loss[i] += output_loss.detach()
            self._per_entry_output_n[i] += target_out.numel()

    @override
    def compute(self) -> dict[str, Float[Tensor, ""]]:
        out: dict[str, Float[Tensor, ""]] = {}
        for i, (suffix, _sources) in enumerate(self._entries):
            class_name = type(self).__name__ + suffix
            per_module = compute_per_module_metrics(
                class_name=f"{class_name}/hidden_acts",
                per_module_sum_mse=self._per_entry_module_sum_mse[i],
                per_module_n_examples=self._per_entry_module_n[i],
            )
            out.update(per_module)
            sum_loss = all_reduce(self._per_entry_output_sum_loss[i], op=ReduceOp.SUM)
            n_examples = all_reduce(self._per_entry_output_n[i].float(), op=ReduceOp.SUM)
            out[f"{class_name}/output_recon"] = sum_loss / n_examples
        return out
