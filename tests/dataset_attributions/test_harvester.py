"""Tests for dataset attribution harvester logic."""

from pathlib import Path

import torch

from spd.dataset_attributions.storage import DatasetAttributionEntry, DatasetAttributionStorage


class TestDatasetAttributionStorage:
    """Tests for DatasetAttributionStorage.

    Matrix structure:
    - Rows (sources): wte tokens [0, vocab_size) + component layers [vocab_size, ...)
    - Cols (targets): component layers [0, n_components) + output tokens [n_components, ...)
    """

    def test_has_source_and_target(self) -> None:
        """Test has_source and has_target methods."""
        # 2 component layers, vocab_size=3
        # Matrix shape: (3 + 2, 2 + 3) = (5, 5)
        storage = DatasetAttributionStorage(
            component_layer_keys=["layer1:0", "layer1:1"],
            vocab_size=3,
            attribution_matrix=torch.zeros(5, 5),
            n_batches_processed=10,
            n_tokens_processed=1000,
            ci_threshold=0.0,
        )

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

    def test_get_attribution(self) -> None:
        """Test get_attribution method."""
        # 2 component layers: a:0, a:1
        # vocab_size=2
        # Matrix shape: (2 + 2, 2 + 2) = (4, 4)
        # Sources: wte:0, wte:1, a:0, a:1 (indices 0, 1, 2, 3)
        # Targets: a:0, a:1, output:0, output:1 (indices 0, 1, 2, 3)
        matrix = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],  # wte:0 -> targets
                [5.0, 6.0, 7.0, 8.0],  # wte:1 -> targets
                [9.0, 10.0, 11.0, 12.0],  # a:0 -> targets
                [13.0, 14.0, 15.0, 16.0],  # a:1 -> targets
            ]
        )
        storage = DatasetAttributionStorage(
            component_layer_keys=["a:0", "a:1"],
            vocab_size=2,
            attribution_matrix=matrix,
            n_batches_processed=1,
            n_tokens_processed=100,
            ci_threshold=0.0,
        )

        # wte:0 -> a:0
        assert storage.get_attribution("wte:0", "a:0") == 1.0
        # wte:1 -> output:1
        assert storage.get_attribution("wte:1", "output:1") == 8.0
        # a:0 -> a:1
        assert storage.get_attribution("a:0", "a:1") == 10.0
        # a:1 -> output:0
        assert storage.get_attribution("a:1", "output:0") == 15.0

    def test_get_top_sources_positive(self) -> None:
        """Test get_top_sources with positive sign."""
        # 2 component layers, vocab_size=2
        # Matrix shape: (4, 4)
        matrix = torch.tensor(
            [
                [1.0, 2.0, 0.0, 0.0],  # wte:0 -> targets
                [5.0, 3.0, 0.0, 0.0],  # wte:1 -> targets
                [2.0, 4.0, 0.0, 0.0],  # a:0 -> targets
                [3.0, 1.0, 0.0, 0.0],  # a:1 -> targets
            ]
        )
        storage = DatasetAttributionStorage(
            component_layer_keys=["a:0", "a:1"],
            vocab_size=2,
            attribution_matrix=matrix,
            n_batches_processed=1,
            n_tokens_processed=100,
            ci_threshold=0.0,
        )

        # Top sources TO a:0 (column 0): wte:0=1.0, wte:1=5.0, a:0=2.0, a:1=3.0
        sources = storage.get_top_sources("a:0", k=2, sign="positive")
        assert len(sources) == 2
        assert sources[0].component_key == "wte:1"
        assert sources[0].value == 5.0
        assert sources[1].component_key == "a:1"
        assert sources[1].value == 3.0

    def test_get_top_sources_negative(self) -> None:
        """Test get_top_sources with negative sign."""
        matrix = torch.tensor(
            [
                [-1.0, 2.0, 0.0, 0.0],
                [-5.0, 3.0, 0.0, 0.0],
                [-2.0, 4.0, 0.0, 0.0],
                [-3.0, 1.0, 0.0, 0.0],
            ]
        )
        storage = DatasetAttributionStorage(
            component_layer_keys=["a:0", "a:1"],
            vocab_size=2,
            attribution_matrix=matrix,
            n_batches_processed=1,
            n_tokens_processed=100,
            ci_threshold=0.0,
        )

        sources = storage.get_top_sources("a:0", k=2, sign="negative")
        assert len(sources) == 2
        # wte:1 has most negative (-5.0), then a:1 (-3.0)
        assert sources[0].component_key == "wte:1"
        assert sources[0].value == -5.0
        assert sources[1].component_key == "a:1"
        assert sources[1].value == -3.0

    def test_get_top_targets(self) -> None:
        """Test get_top_targets method."""
        # Matrix shape: (4, 4)
        # Row 2 (a:0) -> targets: [2.0, 4.0, 1.0, 3.0]
        matrix = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [2.0, 4.0, 1.0, 3.0],  # a:0 -> targets
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        storage = DatasetAttributionStorage(
            component_layer_keys=["a:0", "a:1"],
            vocab_size=2,
            attribution_matrix=matrix,
            n_batches_processed=1,
            n_tokens_processed=100,
            ci_threshold=0.0,
        )

        targets = storage.get_top_targets("a:0", k=2, sign="positive")
        assert len(targets) == 2
        # a:1 (4.0), output:1 (3.0)
        assert targets[0].component_key == "a:1"
        assert targets[0].value == 4.0
        assert targets[1].component_key == "output:1"
        assert targets[1].value == 3.0

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test save and load roundtrip."""
        # 2 component layers, vocab_size=3
        # Matrix shape: (5, 5)
        original = DatasetAttributionStorage(
            component_layer_keys=["layer:0", "layer:1"],
            vocab_size=3,
            attribution_matrix=torch.randn(5, 5),
            n_batches_processed=100,
            n_tokens_processed=10000,
            ci_threshold=0.01,
        )

        path = tmp_path / "test_attributions.pt"
        original.save(path)

        loaded = DatasetAttributionStorage.load(path)

        assert loaded.component_layer_keys == original.component_layer_keys
        assert loaded.vocab_size == original.vocab_size
        assert loaded.n_batches_processed == original.n_batches_processed
        assert loaded.n_tokens_processed == original.n_tokens_processed
        assert loaded.ci_threshold == original.ci_threshold
        assert torch.allclose(loaded.attribution_matrix, original.attribution_matrix)


class TestDatasetAttributionEntry:
    """Tests for DatasetAttributionEntry dataclass."""

    def test_entry_creation(self) -> None:
        """Test creating an entry."""
        entry = DatasetAttributionEntry(
            component_key="h.0.mlp.c_fc:5",
            layer="h.0.mlp.c_fc",
            component_idx=5,
            value=0.123,
        )
        assert entry.component_key == "h.0.mlp.c_fc:5"
        assert entry.layer == "h.0.mlp.c_fc"
        assert entry.component_idx == 5
        assert entry.value == 0.123
