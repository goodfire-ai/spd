import torch

from spd.losses import _calc_tensors_squared_error


class TestCalcFaithfulnessLoss:
    # Actually testing _calc_tensors_squared_error. calc_faithfulness_loss should fail hard in most
    # cases, and testing it would require lots of mocking the way it is currently written. Though
    # now that we don't normalize _calc_tensors_squared_error by the number of parameters, having a
    # test would be helpful.
    def test_calc_faithfulness_loss_single_instance_single_param(self):
        V = torch.ones(2, 3)
        U = torch.ones(3, 2)
        spd_params = {"layer1": U.T @ V.T}
        target_params = {"layer1": torch.tensor([[1.0, 1.0], [1.0, 1.0]])}

        result = _calc_tensors_squared_error(
            params1=target_params,
            params2=spd_params,
            device="cpu",
        )

        # V: [2, 3], U: [3, 2], both filled with ones
        # U^TV^T: [[3, 3], [3, 3]]
        # (U^TV^T - pretrained_weights)^2: [[4, 4], [4, 4]]
        # Sum: 16
        expected = torch.tensor(16.0)
        assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

    def test_calc_faithfulness_loss_single_instance_multiple_params(self):
        Vs = [torch.ones(3, 3), torch.ones(2, 3)]
        Us = [torch.ones(3, 2), torch.ones(3, 3)]
        target_params = {
            "layer1": torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
            "layer2": torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        }
        spd_params = {
            "layer1": Us[0].T @ Vs[0].T,
            "layer2": Us[1].T @ Vs[1].T,
        }
        result = _calc_tensors_squared_error(
            params1=target_params,
            params2=spd_params,
            device="cpu",
        )

        # First layer: UV1: [[3, 3, 3], [3, 3, 3]], diff^2: [[1, 1, 1], [1, 1, 1]]
        # Second layer: UV2: [[3, 3], [3, 3], [3, 3]], diff^2: [[4, 4], [4, 4], [4, 4]]
        # Add together 24 + 6 = 30
        expected = torch.tensor(30.0)
        assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"
