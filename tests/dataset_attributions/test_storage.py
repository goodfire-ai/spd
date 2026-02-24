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


def _make_storage(
    seed: int = 0, n_batches: int = 10, n_tokens: int = 640
) -> DatasetAttributionStorage:
    """Build storage for test topology.

    Sources by target:
        "0.glu.up": ["embed"]             -> embed edge (C0, VOCAB_SIZE)
        "1.glu.up": ["embed", "0.glu.up"] -> embed edge (C1, VOCAB_SIZE) + regular (C1, C0)
        "output":   ["0.glu.up", "1.glu.up"] -> unembed (D_MODEL, C0), (D_MODEL, C1)
        "output":   ["embed"]             -> embed_unembed (D_MODEL, VOCAB_SIZE)
    """
    g = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=g)

    return DatasetAttributionStorage(
        regular_attr={LAYER_1: {LAYER_0: rand(C1, C0)}},
        regular_attr_abs={LAYER_1: {LAYER_0: rand(C1, C0)}},
        embed_attr={LAYER_0: rand(C0, VOCAB_SIZE), LAYER_1: rand(C1, VOCAB_SIZE)},
        embed_attr_abs={LAYER_0: rand(C0, VOCAB_SIZE), LAYER_1: rand(C1, VOCAB_SIZE)},
        unembed_attr={LAYER_0: rand(D_MODEL, C0), LAYER_1: rand(D_MODEL, C1)},
        embed_unembed_attr=rand(D_MODEL, VOCAB_SIZE),
        w_unembed=rand(D_MODEL, VOCAB_SIZE),
        vocab_size=VOCAB_SIZE,
        ci_threshold=1e-6,
        n_batches_processed=n_batches,
        n_tokens_processed=n_tokens,
    )


class TestNComponents:
    def test_counts_all_target_layers(self):
        storage = _make_storage()
        # LAYER_0 is only in embed_attr, LAYER_1 is in both — both count
        assert storage.n_components == C0 + C1


class TestGetTopSources:
    def test_component_target_returns_entries(self):
        storage = _make_storage()
        results = storage.get_top_sources(f"{LAYER_1}:0", k=5, sign="positive", metric="attr")
        assert all(r.value > 0 for r in results)
        assert len(results) <= 5

    def test_component_target_includes_embed(self):
        storage = _make_storage()
        results = storage.get_top_sources(f"{LAYER_1}:0", k=20, sign="positive", metric="attr")
        layers = {r.layer for r in results}
        # Should include both component and embed sources
        assert "embed" in layers or LAYER_0 in layers

    def test_output_target(self):
        storage = _make_storage()
        results = storage.get_top_sources("output:0", k=5, sign="positive", metric="attr")
        assert len(results) <= 5

    def test_output_target_attr_abs_returns_empty(self):
        storage = _make_storage()
        results = storage.get_top_sources("output:0", k=5, sign="positive", metric="attr_abs")
        assert results == []

    def test_target_only_in_embed_attr(self):
        storage = _make_storage()
        # LAYER_0 is only in embed_attr, not in regular_attr
        results = storage.get_top_sources(f"{LAYER_0}:0", k=5, sign="positive", metric="attr")
        assert len(results) <= 5
        assert all(r.layer == "embed" for r in results)

    def test_attr_abs_metric(self):
        storage = _make_storage()
        results = storage.get_top_sources(f"{LAYER_1}:0", k=5, sign="positive", metric="attr_abs")
        assert len(results) <= 5


class TestGetTopTargets:
    def test_component_source(self):
        storage = _make_storage()
        results = storage.get_top_targets(
            f"{LAYER_0}:0", k=5, sign="positive", metric="attr", include_outputs=False
        )
        assert len(results) <= 5
        assert all(r.value > 0 for r in results)

    def test_embed_source(self):
        storage = _make_storage()
        results = storage.get_top_targets(
            "embed:0", k=5, sign="positive", metric="attr", include_outputs=False
        )
        assert len(results) <= 5

    def test_include_outputs(self):
        storage = _make_storage()
        results = storage.get_top_targets(f"{LAYER_0}:0", k=20, sign="positive", metric="attr")
        assert len(results) > 0

    def test_embed_source_with_outputs(self):
        storage = _make_storage()
        results = storage.get_top_targets("embed:0", k=20, sign="positive", metric="attr")
        assert len(results) > 0

    def test_attr_abs_skips_output_targets(self):
        storage = _make_storage()
        results = storage.get_top_targets(f"{LAYER_0}:0", k=20, sign="positive", metric="attr_abs")
        assert all(r.layer != "output" for r in results)


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

        # Check regular_attr roundtrip
        for target in original.regular_attr:
            for source in original.regular_attr[target]:
                torch.testing.assert_close(
                    loaded.regular_attr[target][source], original.regular_attr[target][source]
                )

        # Check embed_attr roundtrip
        for target in original.embed_attr:
            torch.testing.assert_close(loaded.embed_attr[target], original.embed_attr[target])

        # Check unembed_attr roundtrip
        for source in original.unembed_attr:
            torch.testing.assert_close(loaded.unembed_attr[source], original.unembed_attr[source])

        # Check embed_unembed_attr roundtrip
        torch.testing.assert_close(loaded.embed_unembed_attr, original.embed_unembed_attr)


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

        # Check regular_attr merge
        for target in s1.regular_attr:
            for source in s1.regular_attr[target]:
                expected = (
                    s1.regular_attr[target][source] * n1 + s2.regular_attr[target][source] * n2
                ) / total
                torch.testing.assert_close(
                    merged.regular_attr[target][source], expected, atol=1e-5, rtol=1e-5
                )

        # Check embed_unembed_attr merge
        expected = (s1.embed_unembed_attr * n1 + s2.embed_unembed_attr * n2) / total
        torch.testing.assert_close(merged.embed_unembed_attr, expected, atol=1e-5, rtol=1e-5)

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
        for target in s1.regular_attr:
            for source in s1.regular_attr[target]:
                expected = (
                    s1.regular_attr[target][source] * n1 + s2.regular_attr[target][source] * n2
                ) / total
                torch.testing.assert_close(
                    merged.regular_attr[target][source], expected, atol=1e-5, rtol=1e-5
                )

    def test_single_file(self, tmp_path: Path):
        original = _make_storage(seed=7, n_batches=10, n_tokens=640)
        path = tmp_path / "rank_0.pt"
        original.save(path)

        merged = DatasetAttributionStorage.merge([path])

        assert merged.n_tokens_processed == original.n_tokens_processed
        for target in original.regular_attr:
            for source in original.regular_attr[target]:
                torch.testing.assert_close(
                    merged.regular_attr[target][source], original.regular_attr[target][source]
                )
