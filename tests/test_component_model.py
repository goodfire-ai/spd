from __future__ import annotations

from typing import cast

import pytest
import torch
from jaxtyping import Float
from torch import Tensor, nn

from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, LinearComponent


class SimpleTestModel(nn.Module):
    """Simple test model with Linear and Embedding layers."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5, bias=True)
        self.linear2 = nn.Linear(5, 3, bias=False)
        self.embedding = nn.Embedding(100, 8)
        self.other_layer = nn.ReLU()  # Non-target layer

    def forward(self, x: Float[Tensor, "..."]):
        return self.linear2(self.linear1(x))


def test_component_replacement_context_manager():
    """Test the _replaced_modules context manager functionality."""
    # Setup: Create ComponentModel with test model
    base_model = SimpleTestModel()
    target_patterns = ["linear1", "linear2", "embedding"]

    comp_model = ComponentModel(
        base_model=base_model,
        target_module_patterns=target_patterns,
        C=4,
        n_ci_mlp_neurons=0,
        pretrained_model_output_attr=None,
    )

    # Get original modules for comparison
    orig_linear1 = base_model.linear1
    orig_linear2 = base_model.linear2
    orig_embedding = base_model.embedding

    comp_linear1: LinearComponent = cast(LinearComponent, comp_model.components["linear1"])
    comp_linear2: LinearComponent = cast(LinearComponent, comp_model.components["linear2"])
    comp_embedding: EmbeddingComponent = cast(
        EmbeddingComponent, comp_model.components["embedding"]
    )
    components_dict: dict[str, LinearComponent | EmbeddingComponent] = {
        "linear1": comp_linear1,
        "linear2": comp_linear2,
        "embedding": comp_embedding,
    }

    # Test 1: Basic replacement and restoration without masks
    assert base_model.linear1 is orig_linear1  # Sanity check

    with comp_model._replaced_modules(components_dict):
        # During context: modules should be replaced with components
        assert base_model.linear1 is comp_linear1
        assert base_model.linear2 is comp_linear2
        assert base_model.embedding is comp_embedding

        # Masks should still be None (not provided)
        assert comp_linear1.mask is None
        assert comp_linear2.mask is None
        assert comp_embedding.mask is None

    # After context: original modules should be restored
    assert base_model.linear1 is orig_linear1
    assert base_model.linear2 is orig_linear2
    assert base_model.embedding is orig_embedding

    # Test 2: Replacement with masks
    masks = {
        "linear1": torch.randn(4),
        "linear2": torch.randn(4),
        "embedding": torch.randn(4),
    }

    with comp_model._replaced_modules(components_dict, masks=masks):
        # During context: modules replaced and masks set
        assert base_model.linear1 is comp_linear1
        assert base_model.linear2 is comp_linear2
        assert base_model.embedding is comp_embedding

        # Masks should be set
        assert comp_linear1.mask is not None
        assert comp_linear2.mask is not None
        assert comp_embedding.mask is not None
        assert torch.equal(comp_linear1.mask, masks["linear1"])
        assert torch.equal(comp_linear2.mask, masks["linear2"])
        assert torch.equal(comp_embedding.mask, masks["embedding"])

    # After context: modules restored and masks cleared
    assert base_model.linear1 is orig_linear1
    assert base_model.linear2 is orig_linear2
    assert base_model.embedding is orig_embedding

    # Masks should be cleared
    assert comp_linear1.mask is None
    assert comp_linear2.mask is None
    assert comp_embedding.mask is None

    # Test 3: Exception handling preserves original state
    # Set masks initially to verify they get cleared even on exception
    components_dict["linear1"].mask = torch.randn(4)
    components_dict["linear2"].mask = torch.randn(4)
    components_dict["embedding"].mask = torch.randn(4)

    with (
        pytest.raises(RuntimeError, match="Test exception"),
        comp_model._replaced_modules(components_dict, masks=masks),
    ):
        # Verify replacement occurred
        assert base_model.linear1 is comp_linear1
        # Raise exception to test cleanup
        raise RuntimeError("Test exception")

    # After exception: original modules should still be restored
    assert base_model.linear1 is orig_linear1
    assert base_model.linear2 is orig_linear2
    assert base_model.embedding is orig_embedding

    # Masks should be cleared even after exception
    assert comp_linear1.mask is None
    assert comp_linear2.mask is None
    assert comp_embedding.mask is None

    # Test 4: Single component replacement
    single_comp_dict = {"linear1": components_dict["linear1"]}
    single_mask = {"linear1": torch.randn(4)}

    with comp_model._replaced_modules(single_comp_dict, masks=single_mask):
        # Only linear1 should be replaced
        assert base_model.linear1 is comp_linear1
        assert base_model.linear2 is orig_linear2  # Unchanged
        assert base_model.embedding is orig_embedding  # Unchanged

        # Only linear1 component should have mask
        assert comp_linear1.mask is not None
        assert torch.equal(comp_linear1.mask, single_mask["linear1"])
        assert comp_linear2.mask is None
        assert comp_embedding.mask is None

    # After context: everything restored/cleared
    assert base_model.linear1 is orig_linear1
    assert comp_linear1.mask is None


def test_component_replacement_nested_contexts():
    """Test nested context manager calls."""
    base_model = SimpleTestModel()
    comp_model = ComponentModel(
        base_model=base_model,
        target_module_patterns=["linear1", "linear2"],
        C=2,
        n_ci_mlp_neurons=0,
        pretrained_model_output_attr=None,
    )

    orig_linear1 = base_model.linear1
    orig_linear2 = base_model.linear2

    comp_linear1: LinearComponent = cast(LinearComponent, comp_model.components["linear1"])
    comp_linear2: LinearComponent = cast(LinearComponent, comp_model.components["linear2"])

    outer_dict = {"linear1": comp_linear1}
    inner_dict = {"linear2": comp_linear2}

    with comp_model._replaced_modules(outer_dict):
        assert base_model.linear1 is comp_linear1
        assert base_model.linear2 is orig_linear2

        with comp_model._replaced_modules(inner_dict):
            # Inner context should also work
            assert base_model.linear1 is comp_linear1  # Still from outer
            assert base_model.linear2 is comp_linear2  # From inner

        # After inner context: inner replacement undone
        assert base_model.linear1 is comp_linear1  # Still from outer
        assert base_model.linear2 is orig_linear2  # Restored

    # After outer context: everything restored
    assert base_model.linear1 is orig_linear1
    assert base_model.linear2 is orig_linear2
