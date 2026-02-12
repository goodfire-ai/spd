"""Tests for ActivationExamplesReservoir."""

import random

import torch

from spd.harvest.reservoir import WINDOW_PAD_SENTINEL, ActivationExamplesReservoir

DEVICE = torch.device("cpu")
N_COMPONENTS = 4
K = 3
WINDOW = 3


def _make_reservoir() -> ActivationExamplesReservoir:
    return ActivationExamplesReservoir(N_COMPONENTS, K, WINDOW, DEVICE)


class TestAdd:
    def test_fills_up_to_k(self):
        r = _make_reservoir()
        comp = 1

        for i in range(K):
            r.add(
                torch.tensor([comp]),
                torch.full((1, WINDOW), i, dtype=torch.long),
                torch.ones(1, WINDOW),
                torch.ones(1, WINDOW) * 0.5,
            )

        assert r.n_items[comp] == K
        assert r.n_seen[comp] == K
        for i in range(K):
            assert r.tokens[comp, i, 0].item() == i

    def test_replacement_after_k(self):
        r = _make_reservoir()
        comp = 0
        random.seed(42)

        n_total = K + 50
        for i in range(n_total):
            r.add(
                torch.tensor([comp]),
                torch.full((1, WINDOW), i, dtype=torch.long),
                torch.ones(1, WINDOW),
                torch.ones(1, WINDOW),
            )

        assert r.n_items[comp] == K
        assert r.n_seen[comp] == n_total

    def test_written_data_matches_input(self):
        r = _make_reservoir()
        tokens = torch.tensor([[7, 8, 9]])
        ci = torch.tensor([[0.1, 0.2, 0.3]])
        acts = torch.tensor([[1.0, 2.0, 3.0]])
        r.add(torch.tensor([2]), tokens, ci, acts)

        assert torch.equal(r.tokens[2, 0], tokens[0])
        assert torch.allclose(r.ci[2, 0], ci[0])
        assert torch.allclose(r.acts[2, 0], acts[0])


class TestMerge:
    def test_merge_combines_underfilled(self):
        r1 = _make_reservoir()
        r2 = _make_reservoir()

        r1.add(
            torch.tensor([0]),
            torch.full((1, WINDOW), 1, dtype=torch.long),
            torch.ones(1, WINDOW),
            torch.ones(1, WINDOW),
        )
        r2.add(
            torch.tensor([0]),
            torch.full((1, WINDOW), 2, dtype=torch.long),
            torch.ones(1, WINDOW),
            torch.ones(1, WINDOW),
        )

        r1.merge(r2)

        assert r1.n_items[0] == 2
        assert r1.n_seen[0] == 2

    def test_merge_weighted_by_n_seen(self):
        torch.manual_seed(0)

        n_trials = 200
        heavy_wins = 0
        for _ in range(n_trials):
            r_heavy = _make_reservoir()
            r_light = _make_reservoir()

            for _ in range(K):
                r_heavy.add(
                    torch.tensor([0]),
                    torch.full((1, WINDOW), 1, dtype=torch.long),
                    torch.ones(1, WINDOW),
                    torch.ones(1, WINDOW),
                )
            r_heavy.n_seen[0] = 1000

            for _ in range(K):
                r_light.add(
                    torch.tensor([0]),
                    torch.full((1, WINDOW), 2, dtype=torch.long),
                    torch.ones(1, WINDOW),
                    torch.ones(1, WINDOW),
                )
            r_light.n_seen[0] = 1

            r_heavy.merge(r_light)
            from_heavy = (r_heavy.tokens[0, :, 0] == 1).sum().item()
            if from_heavy == K:
                heavy_wins += 1

        assert heavy_wins > n_trials * 0.8

    def test_merge_n_seen_sums(self):
        r1 = _make_reservoir()
        r2 = _make_reservoir()

        for i in range(K + 5):
            r1.add(
                torch.tensor([0]),
                torch.full((1, WINDOW), i % 10, dtype=torch.long),
                torch.ones(1, WINDOW),
                torch.ones(1, WINDOW),
            )
        for i in range(K + 3):
            r2.add(
                torch.tensor([0]),
                torch.full((1, WINDOW), i % 10, dtype=torch.long),
                torch.ones(1, WINDOW),
                torch.ones(1, WINDOW),
            )

        total = r1.n_seen[0].item() + r2.n_seen[0].item()
        r1.merge(r2)
        assert r1.n_seen[0] == total
        assert r1.n_items[0] == K


class TestExamples:
    def test_yields_correct_items(self):
        r = _make_reservoir()
        for i in range(2):
            r.add(
                torch.tensor([0]),
                torch.full((1, WINDOW), i + 10, dtype=torch.long),
                torch.ones(1, WINDOW) * (i + 1) * 0.1,
                torch.ones(1, WINDOW) * (i + 1),
            )

        examples = list(r.examples(0))
        assert len(examples) == 2
        toks_0, ci_0, acts_0 = examples[0]
        assert toks_0.tolist() == [10, 10, 10]
        assert torch.allclose(ci_0, torch.tensor([0.1, 0.1, 0.1]))
        assert torch.allclose(acts_0, torch.tensor([1.0, 1.0, 1.0]))

    def test_filters_sentinels(self):
        r = _make_reservoir()
        r.tokens[0, 0] = torch.tensor([WINDOW_PAD_SENTINEL, 5, 6])
        r.ci[0, 0] = torch.tensor([0.0, 0.8, 0.9])
        r.acts[0, 0] = torch.tensor([0.0, 1.0, 2.0])
        r.n_items[0] = 1
        r.n_seen[0] = 1

        examples = list(r.examples(0))
        assert len(examples) == 1
        toks, ci, acts = examples[0]
        assert toks.tolist() == [5, 6]
        assert torch.allclose(ci, torch.tensor([0.8, 0.9]))
        assert torch.allclose(acts, torch.tensor([1.0, 2.0]))

    def test_empty_component_yields_nothing(self):
        r = _make_reservoir()
        assert list(r.examples(0)) == []


class TestStateDictRoundtrip:
    def test_roundtrip_preserves_data(self):
        r = _make_reservoir()
        for i in range(2):
            r.add(
                torch.tensor([1]),
                torch.full((1, WINDOW), i + 5, dtype=torch.long),
                torch.ones(1, WINDOW) * 0.5,
                torch.ones(1, WINDOW) * 2.0,
            )

        sd = r.state_dict()
        restored = ActivationExamplesReservoir.from_state_dict(sd)

        assert restored.k == r.k
        assert restored.window == r.window
        assert torch.equal(restored.tokens, r.tokens)
        assert torch.equal(restored.ci, r.ci)
        assert torch.equal(restored.acts, r.acts)
        assert torch.equal(restored.n_items, r.n_items)
        assert torch.equal(restored.n_seen, r.n_seen)

    def test_state_dict_on_cpu(self):
        r = _make_reservoir()
        r.add(
            torch.tensor([0]),
            torch.full((1, WINDOW), 1, dtype=torch.long),
            torch.ones(1, WINDOW),
            torch.ones(1, WINDOW),
        )

        sd = r.state_dict()
        assert sd["tokens"].device == torch.device("cpu")
        assert sd["n_items"].device == torch.device("cpu")
