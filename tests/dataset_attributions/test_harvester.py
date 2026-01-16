"""Tests for dataset attribution harvester logic."""

from pathlib import Path

import torch

from spd.dataset_attributions.storage import DatasetAttributionEntry, DatasetAttributionStorage


class TestDatasetAttributionStorage:
    """Tests for DatasetAttributionStorage."""

    def test_has_component(self) -> None:
        """Test has_component method."""
        storage = DatasetAttributionStorage(
            component_keys=["layer1:0", "layer1:1", "layer2:0"],
            attribution_matrix=torch.zeros(3, 3),
            n_batches_processed=10,
            n_tokens_processed=1000,
            ci_threshold=0.0,
        )

        assert storage.has_component("layer1:0")
        assert storage.has_component("layer1:1")
        assert storage.has_component("layer2:0")
        assert not storage.has_component("layer1:2")
        assert not storage.has_component("nonexistent:0")

    def test_get_attribution(self) -> None:
        """Test get_attribution method."""
        matrix = torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [3.0, 0.0, 4.0],
                [5.0, 6.0, 0.0],
            ]
        )
        storage = DatasetAttributionStorage(
            component_keys=["a:0", "a:1", "b:0"],
            attribution_matrix=matrix,
            n_batches_processed=1,
            n_tokens_processed=100,
            ci_threshold=0.0,
        )

        # Test attribution from a:0 to a:1
        assert storage.get_attribution("a:0", "a:1") == 1.0
        # Test attribution from a:1 to b:0
        assert storage.get_attribution("a:1", "b:0") == 4.0
        # Test attribution from b:0 to a:0
        assert storage.get_attribution("b:0", "a:0") == 5.0

    def test_get_top_sources_positive(self) -> None:
        """Test get_top_sources with positive sign."""
        # Matrix: attribution_matrix[src, tgt]
        # Attributions TO component "c:0" (index 2) are column 2: [2.0, 4.0, 0.0]
        matrix = torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [3.0, 0.0, 4.0],
                [5.0, 6.0, 0.0],
            ]
        )
        storage = DatasetAttributionStorage(
            component_keys=["a:0", "b:0", "c:0"],
            attribution_matrix=matrix,
            n_batches_processed=1,
            n_tokens_processed=100,
            ci_threshold=0.0,
        )

        sources = storage.get_top_sources("c:0", k=2, sign="positive")
        assert len(sources) == 2
        # b:0 has highest attribution (4.0), then a:0 (2.0)
        assert sources[0].component_key == "b:0"
        assert sources[0].value == 4.0
        assert sources[1].component_key == "a:0"
        assert sources[1].value == 2.0

    def test_get_top_sources_negative(self) -> None:
        """Test get_top_sources with negative sign."""
        matrix = torch.tensor(
            [
                [0.0, 1.0, -2.0],
                [3.0, 0.0, -4.0],
                [5.0, 6.0, 0.0],
            ]
        )
        storage = DatasetAttributionStorage(
            component_keys=["a:0", "b:0", "c:0"],
            attribution_matrix=matrix,
            n_batches_processed=1,
            n_tokens_processed=100,
            ci_threshold=0.0,
        )

        sources = storage.get_top_sources("c:0", k=2, sign="negative")
        assert len(sources) == 2
        # b:0 has most negative (-4.0), then a:0 (-2.0)
        assert sources[0].component_key == "b:0"
        assert sources[0].value == -4.0
        assert sources[1].component_key == "a:0"
        assert sources[1].value == -2.0

    def test_get_top_targets(self) -> None:
        """Test get_top_targets method."""
        # Matrix: attribution_matrix[src, tgt]
        # Attributions FROM component "a:0" (index 0) are row 0: [0.0, 1.0, 2.0]
        matrix = torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [3.0, 0.0, 4.0],
                [5.0, 6.0, 0.0],
            ]
        )
        storage = DatasetAttributionStorage(
            component_keys=["a:0", "b:0", "c:0"],
            attribution_matrix=matrix,
            n_batches_processed=1,
            n_tokens_processed=100,
            ci_threshold=0.0,
        )

        targets = storage.get_top_targets("a:0", k=2, sign="positive")
        assert len(targets) == 2
        # c:0 has highest (2.0), then b:0 (1.0)
        assert targets[0].component_key == "c:0"
        assert targets[0].value == 2.0
        assert targets[1].component_key == "b:0"
        assert targets[1].value == 1.0

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test save and load roundtrip."""
        original = DatasetAttributionStorage(
            component_keys=["wte:0", "layer:0", "layer:1", "output:0"],
            attribution_matrix=torch.randn(4, 4),
            n_batches_processed=100,
            n_tokens_processed=10000,
            ci_threshold=0.01,
        )

        path = tmp_path / "test_attributions.pt"
        original.save(path)

        loaded = DatasetAttributionStorage.load(path)

        assert loaded.component_keys == original.component_keys
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
