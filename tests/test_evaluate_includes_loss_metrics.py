from unittest.mock import Mock, patch

import torch

from spd.configs import Config, EvalMetricConfig
from spd.eval import evaluate
from spd.models.component_model import ComponentModel


def test_evaluate_includes_loss_terms_metric_when_enabled():
    # Minimal config enabling default loss metrics
    config = Mock(spec=Config)
    config.eval_metrics = []
    config.sigmoid_type = "leaky_hard"
    config.sampling = "continuous"
    config.include_loss_metrics_in_eval = True

    # Component model with minimal API
    model = Mock(spec=ComponentModel)
    model.components = {"layer": Mock()}

    # Iterator yielding a couple of dummy batches
    def iterator():
        for _ in range(2):
            yield torch.zeros(2, 3, dtype=torch.long)

    # Mock pre_forward_cache forward and calc_causal_importances
    def fake_pre_forward(batch, mode, module_names):  # type: ignore[unused-argument]
        return torch.zeros(2, 3, 5), {"layer": torch.zeros(2, 3, 7)}

    model.__call__.side_effect = fake_pre_forward  # type: ignore[attr-defined]
    model.calc_causal_importances.return_value = (
        {"layer": torch.zeros(2, 3, 4)},
        {"layer": torch.zeros(2, 3, 4)},
    )

    with patch("spd.eval.calculate_losses", autospec=True) as mock_calc_losses:
        # Return deterministic loss terms so we can check presence
        mock_calc_losses.return_value = (torch.tensor(2.0), {"faithfulness": 2.0, "total": 2.0})

        out = evaluate(
            model=model,
            eval_iterator=iterator(),
            device="cpu",
            config=config,  # type: ignore[arg-type]
            run_slow=False,
            n_steps=2,
        )

    assert "loss/faithfulness" in out
    assert "loss/total_weighted" in out


def test_evaluate_skips_faithfulness_metric_when_loss_metrics_enabled():
    # If user configured FaithfulnessLoss explicitly and default loss metrics are enabled,
    # we skip the explicit metric to prevent key collisions.
    cfg = Mock(spec=Config)
    cfg.eval_metrics = [EvalMetricConfig(classname="FaithfulnessLoss", extra_init_kwargs={})]
    cfg.sigmoid_type = "leaky_hard"
    cfg.sampling = "continuous"
    cfg.include_loss_metrics_in_eval = True

    model = Mock(spec=ComponentModel)
    model.components = {"layer": Mock()}

    def iterator():
        for _ in range(1):
            yield torch.zeros(2, 3, dtype=torch.long)

    def fake_pre_forward(batch, mode, module_names):  # type: ignore[unused-argument]
        return torch.zeros(2, 3, 5), {"layer": torch.zeros(2, 3, 7)}

    model.__call__.side_effect = fake_pre_forward  # type: ignore[attr-defined]
    model.calc_causal_importances.return_value = (
        {"layer": torch.zeros(2, 3, 4)},
        {"layer": torch.zeros(2, 3, 4)},
    )

    with patch("spd.eval.calculate_losses", autospec=True) as mock_calc_losses:
        mock_calc_losses.return_value = (torch.tensor(1.0), {"faithfulness": 1.0, "total": 1.0})

        out = evaluate(
            model=model,
            eval_iterator=iterator(),
            device="cpu",
            config=cfg,  # type: ignore[arg-type]
            run_slow=False,
            n_steps=1,
        )

    # Should still include the aggregated key, but not duplicate
    assert "loss/faithfulness" in out
