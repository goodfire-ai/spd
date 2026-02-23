"""Tests for DatasetAttributionStorage."""

from pathlib import Path

import torch
from torch import Tensor

from spd.dataset_attributions.storage import DatasetAttributionStorage

VOCAB_SIZE = 4
D_MODEL = 4
LAYER_0 = "0.glu.up"
LAYER_1 = "1.glu.up"
C0 = 3  # components in layer 0
C1 = 2  # components in layer 1


def _make_attr_dict(seed: int = 0) -> dict[str, dict[str, Tensor]]:
    """Build attr dict for the test topology.

    Sources by target:
        "0.glu.up": ["embed"]             -> shape (C0, VOCAB_SIZE)
        "1.glu.up": ["embed", "0.glu.up"] -> shape (C1, VOCAB_SIZE), (C1, C0)
        "output":   ["0.glu.up", "1.glu.up"] -> shape (D_MODEL, C0), (D_MODEL, C1)
    """
    g = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=g)

    return {
        LAYER_0: {"embed": rand(C0, VOCAB_SIZE)},
        LAYER_1: {"embed": rand(C1, VOCAB_SIZE), LAYER_0: rand(C1, C0)},
        "output": {LAYER_0: rand(D_MODEL, C0), LAYER_1: rand(D_MODEL, C1)},
    }


def _make_storage(
    seed: int = 0, n_batches: int = 10, n_tokens: int = 640
) -> DatasetAttributionStorage:
    return DatasetAttributionStorage(
        attr=_make_attr_dict(seed),
        attr_abs=_make_attr_dict(seed + 100),
        mean_squared_attr=_make_attr_dict(seed + 200),
        vocab_size=VOCAB_SIZE,
        ci_threshold=1e-6,
        n_batches_processed=n_batches,
        n_tokens_processed=n_tokens,
    )


class TestNComponents:
    def test_counts_non_output_targets(self):
        storage = _make_storage()
        assert storage.n_components == C0 + C1


class TestHasSource:
    def test_embed_token(self):
        storage = _make_storage()
        assert storage.has_source("embed:0")
        assert storage.has_source(f"embed:{VOCAB_SIZE - 1}")

    def test_embed_oob(self):
        storage = _make_storage()
        assert not storage.has_source(f"embed:{VOCAB_SIZE}")
        assert not storage.has_source("embed:-1")

    def test_component_source(self):
        storage = _make_storage()
        assert storage.has_source(f"{LAYER_0}:0")
        assert storage.has_source(f"{LAYER_0}:{C0 - 1}")

    def test_component_source_oob(self):
        storage = _make_storage()
        assert not storage.has_source(f"{LAYER_0}:{C0}")

    def test_output_never_source(self):
        storage = _make_storage()
        assert not storage.has_source("output:0")

    def test_layer_not_present(self):
        storage = _make_storage()
        assert not storage.has_source("nonexistent:0")


class TestHasTarget:
    def test_component_target(self):
        storage = _make_storage()
        assert storage.has_target(f"{LAYER_0}:0")
        assert storage.has_target(f"{LAYER_1}:{C1 - 1}")

    def test_component_target_oob(self):
        storage = _make_storage()
        assert not storage.has_target(f"{LAYER_0}:{C0}")
        assert not storage.has_target(f"{LAYER_1}:{C1}")

    def test_output_target(self):
        storage = _make_storage()
        assert storage.has_target("output:0")
        assert storage.has_target(f"output:{VOCAB_SIZE - 1}")

    def test_output_target_oob(self):
        storage = _make_storage()
        assert not storage.has_target(f"output:{VOCAB_SIZE}")

    def test_embed_never_target(self):
        storage = _make_storage()
        assert not storage.has_target("embed:0")

    def test_layer_not_present(self):
        storage = _make_storage()
        assert not storage.has_target("nonexistent:0")


class TestSaveLoad:
    def test_roundtrip(self, tmp_path: Path):
        original = _make_storage()
        path = tmp_path / "attrs.pt"
        original.save(path)

        loaded = DatasetAttributionStorage.load(path)

        assert loaded.vocab_size == original.vocab_size
        assert loaded.ci_threshold == original.ci_threshold
        assert loaded.n_batches_processed == original.n_batches_processed
        assert loaded.n_tokens_processed == original.n_tokens_processed
        assert loaded.n_components == original.n_components

        for attr_name in ("attr", "attr_abs", "mean_squared_attr"):
            orig_dict = getattr(original, attr_name)
            load_dict = getattr(loaded, attr_name)
            assert orig_dict.keys() == load_dict.keys()
            for target in orig_dict:
                assert orig_dict[target].keys() == load_dict[target].keys()
                for source in orig_dict[target]:
                    torch.testing.assert_close(load_dict[target][source], orig_dict[target][source])


class TestMerge:
    def test_two_workers_weighted_average(self, tmp_path: Path):
        s1 = _make_storage(seed=0, n_batches=5, n_tokens=320)
        s2 = _make_storage(seed=42, n_batches=5, n_tokens=320)

        p1 = tmp_path / "rank_0.pt"
        p2 = tmp_path / "rank_1.pt"
        s1.save(p1)
        s2.save(p2)

        merged = DatasetAttributionStorage.merge([p1, p2])

        assert merged.n_batches_processed == 10
        assert merged.n_tokens_processed == 640
        assert merged.vocab_size == VOCAB_SIZE
        assert merged.ci_threshold == s1.ci_threshold

        n1, n2 = s1.n_tokens_processed, s2.n_tokens_processed
        total = n1 + n2
        for target in s1.attr:
            for source in s1.attr[target]:
                expected = (s1.attr[target][source] * n1 + s2.attr[target][source] * n2) / total
                torch.testing.assert_close(
                    merged.attr[target][source], expected, atol=1e-5, rtol=1e-5
                )

    def test_unequal_token_counts(self, tmp_path: Path):
        s1 = _make_storage(seed=0, n_batches=3, n_tokens=192)
        s2 = _make_storage(seed=42, n_batches=7, n_tokens=448)

        p1 = tmp_path / "rank_0.pt"
        p2 = tmp_path / "rank_1.pt"
        s1.save(p1)
        s2.save(p2)

        merged = DatasetAttributionStorage.merge([p1, p2])

        assert merged.n_tokens_processed == 640
        assert merged.n_batches_processed == 10

        n1, n2 = s1.n_tokens_processed, s2.n_tokens_processed
        total = n1 + n2
        for target in s1.attr:
            for source in s1.attr[target]:
                expected = (s1.attr[target][source] * n1 + s2.attr[target][source] * n2) / total
                torch.testing.assert_close(
                    merged.attr[target][source], expected, atol=1e-5, rtol=1e-5
                )

    def test_single_file(self, tmp_path: Path):
        original = _make_storage(seed=7, n_batches=10, n_tokens=640)
        path = tmp_path / "rank_0.pt"
        original.save(path)

        merged = DatasetAttributionStorage.merge([path])

        assert merged.n_tokens_processed == original.n_tokens_processed
        for target in original.attr:
            for source in original.attr[target]:
                torch.testing.assert_close(
                    merged.attr[target][source], original.attr[target][source]
                )
