from unittest.mock import Mock, patch

import torch
from jaxtyping import Float
from torch import Tensor

from spd.configs import Config
from spd.eval import LossTermsMetric
from spd.models.component_model import ComponentModel


def _make_ci(batch: int = 2, seq: int = 3, C: int = 4):
    return {"layer": torch.zeros(batch, seq, C)}


def test_loss_terms_metric_aggregates_means():
    # Mocks
    model = Mock(spec=ComponentModel)
    model.components = {"layer": Mock()}

    # calc_weight_deltas is used inside metric; we'll let it return a dummy dict via real impl
    # We'll patch calculate_losses to control outputs deterministically
    config = Mock(spec=Config)
    config.sigmoid_type = "leaky_hard"
    config.sampling = "continuous"
    config.output_loss_type = "kl"
    config.faithfulness_coeff = 1.0
    config.recon_coeff = None
    config.stochastic_recon_coeff = None
    config.recon_layerwise_coeff = None
    config.stochastic_recon_layerwise_coeff = None
    config.importance_minimality_coeff = 0.1
    config.schatten_coeff = None
    config.out_recon_coeff = None
    config.embedding_recon_coeff = None

    # Prepare model forward stubs for pre_forward_cache path
    def fake_pre_forward(batch, mode, module_names):  # type: ignore[unused-argument]
        return torch.zeros(2, 3, 5), {"layer": torch.zeros(2, 3, 7)}

    model.__call__.side_effect = fake_pre_forward  # type: ignore[attr-defined]

    metric = LossTermsMetric(model, config)  # type: ignore[arg-type]

    batch = torch.zeros(2, 3, dtype=torch.long)
    target_out: Float[Tensor, "... vocab"] = torch.zeros(2, 3, 5)
    ci = _make_ci()

    with patch("spd.eval.calculate_losses", autospec=True) as mock_calc_losses:
        # Two batches with different values
        mock_calc_losses.side_effect = [
            (torch.tensor(2.0), {"faithfulness": 2.0, "importance_minimality": 1.0, "total": 2.0}),
            (torch.tensor(4.0), {"faithfulness": 4.0, "importance_minimality": 3.0, "total": 4.0}),
        ]

        metric.watch_batch(batch=batch, target_out=target_out, ci=ci)
        metric.watch_batch(batch=batch, target_out=target_out, ci=ci)

        out = metric.compute()

    # Means across 2 batches
    assert out["loss/faithfulness"] == 3.0
    assert out["loss/importance_minimality"] == 2.0
    assert out["loss/total_weighted"] == 3.0
