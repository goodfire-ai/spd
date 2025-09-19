"""Test identity insertion functionality."""

from typing import override

import torch
import torch.nn as nn
from torch.testing import assert_close

from spd.utils.identity_insertion import insert_identity_operations_


class SimpleModel(nn.Module):
    """Simple test model with multiple linear layers."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(100, d_model)
        self.layer1 = nn.Linear(d_model, d_model)
        self.layer2 = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, 100)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return self.output(x)


def test_identity_insertion_inserts_identity_layers():
    model = SimpleModel(d_model=32).to("cpu")
    model.eval()

    insert_identity_operations_(
        target_model=model,
        identity_patterns=["layer1", "layer2"],
        device="cpu",
    )

    assert hasattr(model.layer1, "pre_identity")
    assert hasattr(model.layer2, "pre_identity")
    assert isinstance(model.layer1.pre_identity, nn.Linear)
    assert isinstance(model.layer2.pre_identity, nn.Linear)
    assert_close(model.layer1.pre_identity.weight, torch.eye(32, device="cpu"))
    assert_close(model.layer2.pre_identity.weight, torch.eye(32, device="cpu"))

    assert not hasattr(model.embedding, "pre_identity")
    assert not hasattr(model.output, "pre_identity")


def test_identity_insertion_preserves_output():
    """Test that inserting identity operations doesn't change model output."""
    torch.manual_seed(42)
    device = "cpu"

    model = SimpleModel(d_model=32).to(device)
    model.eval()

    input_ids = torch.randint(0, 100, (2, 10), device=device)

    with torch.no_grad():
        original_output = model(input_ids)

    insert_identity_operations_(model, identity_patterns=["layer1"], device=device)

    with torch.no_grad():
        new_output = model(input_ids)

    assert_close(original_output, new_output, atol=1e-6, rtol=1e-6)


def test_identity_insertion_uses_correct_dims():
    """Test identity insertion with layers of different dimensions."""
    torch.manual_seed(42)
    device = "cpu"

    class VaryingDimModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 64)
            self.layer1 = nn.Linear(64, 128)
            self.layer2 = nn.Linear(128, 256)
            self.output = nn.Linear(256, 100)

        @override
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embedding(x)
            x = self.layer1(x)
            x = torch.relu(x)
            x = self.layer2(x)
            return self.output(x)

    model = VaryingDimModel().to(device)
    model.eval()

    # Insert identity only before layer1 (which takes 64-dim input)
    insert_identity_operations_(
        target_model=model,
        identity_patterns=["layer1"],
        device=device,
    )

    # Check that identity has correct dimension
    assert hasattr(model.layer1, "pre_identity")
    assert isinstance(model.layer1.pre_identity, nn.Linear)
    assert model.layer1.pre_identity.weight.shape == (64, 64)

    # layer2 should not have identity
    assert not hasattr(model.layer2, "pre_identity")


def test_identity_insertion_empty_patterns():
    """Test that empty patterns don't break anything."""
    model = SimpleModel().to("cpu")

    # No patterns should result in no modifications
    insert_identity_operations_(
        target_model=model,
        identity_patterns=[],
        device="cpu",
    )

    # No identity layers should be added
    assert not hasattr(model.embedding, "pre_identity")
    assert not hasattr(model.layer1, "pre_identity")
    assert not hasattr(model.layer2, "pre_identity")
    assert not hasattr(model.output, "pre_identity")
