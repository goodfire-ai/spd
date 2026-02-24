"""Tests for dataset attribution harvester logic."""

from pathlib import Path

import torch

from spd.dataset_attributions.storage import DatasetAttributionStorage


def _make_storage(
    n_components: int = 2,
    vocab_size: int = 3,
    d_model: int = 4,
    source_to_component: torch.Tensor | None = None,
    source_to_out_residual: torch.Tensor | None = None,
) -> DatasetAttributionStorage:
    """Helper to create storage with default values."""
    n_sources = vocab_size + n_components
    if source_to_component is None:
        source_to_component = torch.zeros(n_sources, n_components)
    if source_to_out_residual is None:
        source_to_out_residual = torch.zeros(n_sources, d_model)

    return DatasetAttributionStorage(
        component_layer_keys=[f"layer1:{i}" for i in range(n_components)],
        vocab_size=vocab_size,
        d_model=d_model,
        source_to_component=source_to_component,
        source_to_out_residual=source_to_out_residual,
        n_batches_processed=10,
        n_tokens_processed=1000,
        ci_threshold=0.0,
    )


class TestDatasetAttributionStorage:
    """Tests for DatasetAttributionStorage.

    Storage structure:
    - source_to_component: (n_sources, n_components) for component target attributions
    - source_to_out_residual: (n_sources, d_model) for output target attributions (via w_unembed)
    """

    def test_has_source_and_target(self) -> None:
        """Test has_source and has_target methods."""
        storage = _make_storage(n_components=2, vocab_size=3)

        # wte tokens can only be sources
        assert storage.has_source("wte:0")
        assert storage.has_source("wte:2")
        assert not storage.has_source("wte:3")  # Out of vocab
        assert not storage.has_target("wte:0")  # wte can't be target

        # Component layers can be both sources and targets
        assert storage.has_source("layer1:0")
        assert storage.has_source("layer1:1")
        assert storage.has_target("layer1:0")
        assert storage.has_target("layer1:1")
        assert not storage.has_source("layer1:2")
        assert not storage.has_target("layer1:2")

        # output tokens can only be targets
        assert storage.has_target("output:0")
        assert storage.has_target("output:2")
        assert not storage.has_target("output:3")  # Out of vocab
        assert not storage.has_source("output:0")  # output can't be source

    def test_get_attribution_component_target(self) -> None:
        """Test get_attribution for component targets (no w_unembed needed)."""
        # 2 component layers: layer1:0, layer1:1
        # vocab_size=2, d_model=4
        # n_sources = 2 + 2 = 4
        # source_to_component shape: (4, 2)
        source_to_component = torch.tensor(
            [
                [1.0, 2.0],  # wte:0 -> components
                [3.0, 4.0],  # wte:1 -> components
                [5.0, 6.0],  # layer1:0 -> components
                [7.0, 8.0],  # layer1:1 -> components
            ]
        )
        storage = _make_storage(
            n_components=2, vocab_size=2, source_to_component=source_to_component
        )

        # wte:0 -> layer1:0
        assert storage.get_attribution("wte:0", "layer1:0", w_unembed=None) == 1.0
        # wte:1 -> layer1:1
        assert storage.get_attribution("wte:1", "layer1:1", w_unembed=None) == 4.0
        # layer1:0 -> layer1:1
        assert storage.get_attribution("layer1:0", "layer1:1", w_unembed=None) == 6.0

    def test_get_attribution_output_target(self) -> None:
        """Test get_attribution for output targets (requires w_unembed)."""
        # source_to_out_residual shape: (4, 4) for n_sources=4, d_model=4
        source_to_out_residual = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],  # wte:0 -> out_residual
                [0.0, 1.0, 0.0, 0.0],  # wte:1 -> out_residual
                [0.0, 0.0, 1.0, 0.0],  # layer1:0 -> out_residual
                [0.0, 0.0, 0.0, 1.0],  # layer1:1 -> out_residual
            ]
        )
        # w_unembed shape: (d_model=4, vocab=2)
        w_unembed = torch.tensor(
            [
                [1.0, 2.0],  # d0 -> outputs
                [3.0, 4.0],  # d1 -> outputs
                [5.0, 6.0],  # d2 -> outputs
                [7.0, 8.0],  # d3 -> outputs
            ]
        )
        storage = _make_storage(
            n_components=2, vocab_size=2, d_model=4, source_to_out_residual=source_to_out_residual
        )

        # wte:0 -> output:0 = out_residual[0] @ w_unembed[:, 0] = [1,0,0,0] @ [1,3,5,7] = 1.0
        assert storage.get_attribution("wte:0", "output:0", w_unembed=w_unembed) == 1.0
        # wte:1 -> output:1 = [0,1,0,0] @ [2,4,6,8] = 4.0
        assert storage.get_attribution("wte:1", "output:1", w_unembed=w_unembed) == 4.0
        # layer1:0 -> output:0 = [0,0,1,0] @ [1,3,5,7] = 5.0
        assert storage.get_attribution("layer1:0", "output:0", w_unembed=w_unembed) == 5.0

    def test_get_top_sources_component_target(self) -> None:
        """Test get_top_sources for component targets."""
        source_to_component = torch.tensor(
            [
                [1.0, 2.0],  # wte:0
                [5.0, 3.0],  # wte:1
                [2.0, 4.0],  # layer1:0
                [3.0, 1.0],  # layer1:1
            ]
        )
        storage = _make_storage(
            n_components=2, vocab_size=2, source_to_component=source_to_component
        )

        # Top sources TO layer1:0 (column 0): wte:0=1.0, wte:1=5.0, layer1:0=2.0, layer1:1=3.0
        sources = storage.get_top_sources("layer1:0", k=2, sign="positive")
        assert len(sources) == 2
        assert sources[0].component_key == "wte:1"
        assert sources[0].value == 5.0
        assert sources[1].component_key == "layer1:1"
        assert sources[1].value == 3.0

    def test_get_top_sources_negative(self) -> None:
        """Test get_top_sources with negative sign."""
        source_to_component = torch.tensor(
            [
                [-1.0, 2.0],
                [-5.0, 3.0],
                [-2.0, 4.0],
                [-3.0, 1.0],
            ]
        )
        storage = _make_storage(
            n_components=2, vocab_size=2, source_to_component=source_to_component
        )

        sources = storage.get_top_sources("layer1:0", k=2, sign="negative")
        assert len(sources) == 2
        # wte:1 has most negative (-5.0), then layer1:1 (-3.0)
        assert sources[0].component_key == "wte:1"
        assert sources[0].value == -5.0
        assert sources[1].component_key == "layer1:1"
        assert sources[1].value == -3.0

    def test_get_top_component_targets(self) -> None:
        """Test get_top_component_targets (no w_unembed needed)."""
        source_to_component = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [2.0, 4.0],  # layer1:0 -> components
                [0.0, 0.0],
            ]
        )
        storage = _make_storage(
            n_components=2, vocab_size=2, source_to_component=source_to_component
        )

        targets = storage.get_top_component_targets("layer1:0", k=2, sign="positive")
        assert len(targets) == 2
        assert targets[0].component_key == "layer1:1"
        assert targets[0].value == 4.0
        assert targets[1].component_key == "layer1:0"
        assert targets[1].value == 2.0

    def test_get_top_targets_with_outputs(self) -> None:
        """Test get_top_targets including outputs (requires w_unembed)."""
        source_to_component = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [2.0, 4.0],  # layer1:0 -> components
                [0.0, 0.0],
            ]
        )
        # Make out_residual attribution that produces high output values
        source_to_out_residual = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],  # layer1:0 -> out_residual (sum=4 per output)
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        # w_unembed that gives output:0=10, output:1=5
        w_unembed = torch.tensor(
            [
                [2.5, 1.25],
                [2.5, 1.25],
                [2.5, 1.25],
                [2.5, 1.25],
            ]
        )
        storage = _make_storage(
            n_components=2,
            vocab_size=2,
            d_model=4,
            source_to_component=source_to_component,
            source_to_out_residual=source_to_out_residual,
        )

        targets = storage.get_top_targets("layer1:0", k=3, sign="positive", w_unembed=w_unembed)
        assert len(targets) == 3
        # output:0 = 10.0, output:1 = 5.0, layer1:1 = 4.0
        assert targets[0].component_key == "output:0"
        assert targets[0].value == 10.0
        assert targets[1].component_key == "output:1"
        assert targets[1].value == 5.0
        assert targets[2].component_key == "layer1:1"
        assert targets[2].value == 4.0

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test save and load roundtrip."""
        n_components = 2
        vocab_size = 3
        d_model = 4
        n_sources = vocab_size + n_components

        original = DatasetAttributionStorage(
            component_layer_keys=["layer:0", "layer:1"],
            vocab_size=vocab_size,
            d_model=d_model,
            source_to_component=torch.randn(n_sources, n_components),
            source_to_out_residual=torch.randn(n_sources, d_model),
            n_batches_processed=100,
            n_tokens_processed=10000,
            ci_threshold=0.01,
        )

        path = tmp_path / "test_attributions.pt"
        original.save(path)

        loaded = DatasetAttributionStorage.load(path)

        assert loaded.component_layer_keys == original.component_layer_keys
        assert loaded.vocab_size == original.vocab_size
        assert loaded.d_model == original.d_model
        assert loaded.n_batches_processed == original.n_batches_processed
        assert loaded.n_tokens_processed == original.n_tokens_processed
        assert loaded.ci_threshold == original.ci_threshold
        assert torch.allclose(loaded.source_to_component, original.source_to_component)
        assert torch.allclose(loaded.source_to_out_residual, original.source_to_out_residual)
